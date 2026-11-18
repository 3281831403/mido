# SPDX-FileCopyrightText: 2016 Ole Martin Bjorndalen <ombdalen@gmail.com>
#
# SPDX-License-Identifier: MIT

from .meta import MetaMessage


class MidiTrack(list):
    @property
    def name(self):
        """Name of the track.

        This will return the name from the first track_name meta
        message in the track, or '' if there is no such message.

        Setting this property will update the name field of the first
        track_name message in the track. If no such message is found,
        one will be added to the beginning of the track with a delta
        time of 0."""
        for message in self:
            if message.type == 'track_name':
                return message.name
        else:
            return ''

    @name.setter
    def name(self, name):
        # Find the first track_name message and modify it.
        for message in self:
            if message.type == 'track_name':
                message.name = name
                return
        else:
            # No track name found, add one.
            self.insert(0, MetaMessage('track_name', name=name, time=0))

    def copy(self):
        return self.__class__(self)

    def __getitem__(self, index_or_slice):
        # Retrieve item from the MidiTrack
        lst = list.__getitem__(self, index_or_slice)
        if isinstance(index_or_slice, int):
            # If an index was provided, return the list element
            return lst
        else:
            # Otherwise, construct a MidiTrack to return.
            # TODO: this make a copy of the list. Is there a better way?
            return self.__class__(lst)

    def __add__(self, other):
        return self.__class__(list.__add__(self, other))

    def __mul__(self, other):
        return self.__class__(list.__mul__(self, other))

    def __repr__(self):
        if len(self) == 0:
            messages = ''
        elif len(self) == 1:
            messages = f'[{self[0]}]'
        else:
            messages = '[\n  {}]'.format(',\n  '.join(repr(m) for m in self))
        return f'{self.__class__.__name__}({messages})'


def _to_abstime(messages, skip_checks=False):
    """Convert messages to absolute time."""
    now = 0
    for msg in messages:
        now += msg.time
        yield msg.copy(skip_checks=skip_checks, time=now)


def _to_abstime_with_ports(track, skip_checks=False):
    """Convert track messages to absolute time.

    Yields (port, message) pairs where port is the midi_port in
    effect for the message in its own track. midi_port meta messages
    are not yielded, but reinserted by merge_tracks() where they are
    needed.
    """
    now = 0
    port = 0
    for msg in track:
        now += msg.time
        if msg.type == 'midi_port':
            port = msg.port
        else:
            yield port, msg.copy(skip_checks=skip_checks, time=now)


def _to_reltime(messages, skip_checks=False):
    """Convert messages to relative time."""
    now = 0
    for msg in messages:
        delta = msg.time - now
        yield msg.copy(skip_checks=skip_checks, time=delta)
        now = msg.time


def fix_end_of_track(messages, skip_checks=False):
    """Remove all end_of_track messages and add one at the end.

    This is used by merge_tracks() and MidiFile.save()."""
    # Accumulated delta time from removed end of track messages.
    # This is added to the next message.
    accum = 0

    for msg in messages:
        if msg.type == 'end_of_track':
            accum += msg.time
        else:
            if accum:
                delta = accum + msg.time
                yield msg.copy(skip_checks=skip_checks, time=delta)
                accum = 0
            else:
                yield msg

    yield MetaMessage('end_of_track', time=accum)


def merge_tracks(tracks, skip_checks=False):
    """Returns a MidiTrack object with all messages from all tracks.

    The messages are returned in playback order with delta times
    as if they were all in one track.

    midi_port meta messages are applied per track before merging and
    reinserted into the merged track, so that every channel message
    is still routed to the port declared by its own track instead of
    being hijacked by port declarations from other tracks.

    Pass skip_checks=True to skip validation of messages before merging.
    This should ONLY be used when the messages in tracks have already
    been validated by mido.checks.
    """
    messages = []
    for track in tracks:
        messages.extend(_to_abstime_with_ports(track, skip_checks=skip_checks))

    messages.sort(key=lambda port_and_msg: port_and_msg[1].time)

    # Reinsert midi_port meta messages so that each channel message is
    # preceded by the port declaration of its own track. (The default
    # port is 0, so files without midi_port messages are unaffected.)
    port = 0
    messages_with_ports = []
    for msg_port, msg in messages:
        if not msg.is_meta and hasattr(msg, 'channel') and msg_port != port:
            port = msg_port
            messages_with_ports.append(
                MetaMessage('midi_port', port=port, time=msg.time))
        messages_with_ports.append(msg)

    return MidiTrack(
        fix_end_of_track(
            _to_reltime(messages_with_ports, skip_checks=skip_checks),
            skip_checks=skip_checks,
        )
    )
