# SPDX-FileCopyrightText: 2016 Ole Martin Bjorndalen <ombdalen@gmail.com>
#
# SPDX-License-Identifier: MIT

"""Tests for legacy Standard MIDI File variants.

Covers files wrapped in a RIFF/RMID container, SMPTE (time code based)
time division and 0xF7 escape events.
"""
import io
import struct

from pytest import approx, raises

from mido.frozen import FrozenSysexEscapeMessage, freeze_message, thaw_message
from mido.messages import Message
from mido.midifiles.meta import MetaMessage
from mido.midifiles.midifiles import (
    MidiFile,
    SysexEscapeMessage,
    decode_division,
    encode_division,
)


def make_smf(division=480, format=0, tracks=()):
    body = b''
    for track in tracks:
        body += b'MTrk' + struct.pack('>L', len(track)) + track
    return (b'MThd' + struct.pack('>L', 6) +
            struct.pack('>hhH', format, len(tracks), division & 0xffff) +
            body)


def make_riff(smf_data, extra=b''):
    payload = (b'RMID' + extra + b'data' +
               struct.pack('<L', len(smf_data)) + smf_data)
    return b'RIFF' + struct.pack('<L', len(payload)) + payload


def load(data):
    return MidiFile(file=io.BytesIO(data))


def save(midifile):
    out = io.BytesIO()
    midifile.save(file=out)
    return out.getvalue()


PPQN_TRACK = bytes([
    0x00, 0x90, 60, 64,        # note_on 60 velocity 64
    0x83, 0x60, 0x80, 60, 64,  # +480 ticks, note_off
    0x00, 0xff, 0x2f, 0x00,    # end_of_track
])
PPQN_SMF = make_smf(480, 0, [PPQN_TRACK])


# RIFF / RMID container.

def test_rmid_wrapped_file():
    midifile = load(make_riff(PPQN_SMF))

    assert midifile.type == 0
    assert midifile.ticks_per_beat == 480
    assert midifile.tracks[0] == [
        Message('note_on', note=60, velocity=64, time=0),
        Message('note_off', note=60, velocity=64, time=480),
        MetaMessage('end_of_track', time=0),
    ]
    assert midifile.length == 0.5


def test_rmid_with_leading_info_chunk():
    # RIFF allows other chunks (such as LIST/INFO) before the data chunk.
    info = b'LIST' + struct.pack('<L', 10) + b'INFOINAM' + b'\x00\x00'
    midifile = load(make_riff(PPQN_SMF, extra=info))

    assert midifile.ticks_per_beat == 480
    assert [msg.type for msg in midifile.tracks[0]] == [
        'note_on', 'note_off', 'end_of_track']
    assert midifile.length == 0.5


def test_riff_that_is_not_rmid_rejected():
    payload = b'WEBP' + b'data' + struct.pack('<L', 0)
    data = b'RIFF' + struct.pack('<L', len(payload)) + payload
    with raises(OSError):
        load(data)


def test_rmid_without_data_chunk_rejected():
    payload = b'RMID' + b'LIST' + struct.pack('<L', 4) + b'INFO'
    data = b'RIFF' + struct.pack('<L', len(payload)) + payload
    with raises(OSError):
        load(data)


# SMPTE time division.

SMPTE_25_TRACK = bytes([
    0x00, 0xff, 0x51, 0x03, 0x0f, 0x42, 0x40,  # tempo 1000000
    0x00, 0x90, 60, 64,                        # note_on at tick 0
    0x87, 0x68, 0x80, 60, 64,                  # +1000 ticks
    0x00, 0xff, 0x2f, 0x00,
])
SMPTE_25_SMF = make_smf(0xE728, 0, [SMPTE_25_TRACK])  # -25 fps, 40 tpf


def test_smpte_division_bytes_decoded():
    assert decode_division(0xE728) == (None, 25, 40)
    assert decode_division(0xE264) == (None, 30, 100)
    assert decode_division(0x01E0) == (480, None, None)


def test_smpte_division_encoded_back_to_same_bytes():
    assert encode_division(25, 40) == 0xE728
    assert encode_division(30, 100) == 0xE264


def test_smpte_division_with_unknown_frame_rate_rejected():
    with raises(ValueError):
        decode_division(0xE064)  # frame rate code -32


def test_smpte_division_with_zero_ticks_per_frame_rejected():
    with raises(ValueError):
        decode_division(0xE700)


def test_smpte_division_without_negative_high_byte_rejected():
    # The high byte of a valid SMPTE division is always negative.
    with raises(ValueError):
        decode_division(0xE100)


def test_smpte_29_97_division_roundtrip():
    assert decode_division(0xE328) == (None, 29.97, 40)
    assert encode_division(29.97, 40) == 0xE328


def test_smpte_division_accepts_unsigned_word():
    # 0xE728 read as an unsigned 16 bit integer is equivalent.
    assert decode_division(0xE728 + 0x10000) == (None, 25, 40)


def test_smpte_25fps_seconds():
    # 25 frames per second * 40 ticks per frame == 1000 ticks per second
    midifile = load(SMPTE_25_SMF)

    assert midifile.ticks_per_beat is None
    assert midifile.frames_per_second == 25
    assert midifile.ticks_per_frame == 40
    assert [msg.time for msg in midifile if msg.type == 'note_off'] == [
        approx(1.0)]
    assert midifile.length == approx(1.0)
    # The tempo map is still read, but SMPTE timing ignores it.
    assert [msg.tempo for msg in midifile if msg.type == 'set_tempo'] == [
        1000000]


def test_smpte_30fps_seconds():
    # 30 frames per second * 100 ticks per frame == 3000 ticks per second
    track = bytes([
        0x00, 0x90, 60, 64,
        0x82, 0x2c, 0x80, 60, 64,  # +300 ticks
        0x00, 0xff, 0x2f, 0x00,
    ])
    midifile = load(make_smf(0xE264, 0, [track]))

    assert [msg.time for msg in midifile if msg.type == 'note_off'] == [
        approx(0.1)]
    assert midifile.length == approx(0.1)


def test_smpte_save_keeps_division_word():
    midifile = load(SMPTE_25_SMF)
    saved = save(midifile)

    assert saved[:4] == b'MThd'
    assert saved[12:14] == b'\xe7\x28'
    assert load(saved).length == approx(1.0)


def test_smpte_save_is_byte_identical():
    midifile = load(SMPTE_25_SMF)
    assert save(midifile) == SMPTE_25_SMF


def test_ppqn_tempo_map_and_division_unchanged():
    # 480 ticks at 500000us plus 480 ticks at 250000us == 0.75 seconds
    track0 = bytes([
        0x00, 0xff, 0x51, 0x03, 0x07, 0xa1, 0x20,  # tempo 500000
        0x00, 0x90, 60, 64,
        0x83, 0x60, 0xff, 0x51, 0x03, 0x03, 0xd0, 0x90,  # tempo 250000
        0x00, 0xff, 0x2f, 0x00,
    ])
    track1 = bytes([
        0x00, 0x90, 62, 64,
        0x87, 0x40, 0x80, 62, 64,
        0x00, 0xff, 0x2f, 0x00,
    ])
    midifile = load(make_smf(480, 1, [track0, track1]))

    assert midifile.ticks_per_beat == 480
    assert midifile.frames_per_second is None
    assert midifile.length == approx(0.75)
    assert save(midifile)[12:14] == b'\x01\xe0'


# 0xF7 escape events.

ESCAPE_TRACK = bytes([
    0x00, 0xf7, 0x03, 0x90, 60, 64,  # escape event with raw note_on bytes
    0x00, 0x90, 62, 64,
    0x83, 0x60, 0x80, 62, 64,
    0x00, 0xff, 0x2f, 0x00,
])
ESCAPE_SMF = make_smf(480, 0, [ESCAPE_TRACK])


def test_escape_event_with_high_bytes_read():
    midifile = load(ESCAPE_SMF)

    assert len(midifile.tracks[0]) == 4
    assert isinstance(midifile.tracks[0][0], SysexEscapeMessage)
    assert midifile.tracks[0][0].data == (0x90, 60, 64)
    assert [msg for msg in midifile.tracks[0]
            if msg.type in ('note_on', 'note_off')] == [
        Message('note_on', note=62, velocity=64, time=0),
        Message('note_off', note=62, velocity=64, time=480),
    ]


def test_escape_event_save_is_byte_identical():
    midifile = load(ESCAPE_SMF)
    saved = save(midifile)

    assert saved == ESCAPE_SMF
    assert save(load(saved)) == saved


def test_sysex_continuation_keeps_f7_event_type():
    track = bytes([
        0x00, 0xf0, 0x04, 0x43, 0x10, 0x4c, 0xf7,  # complete sysex
        0x00, 0xf7, 0x03, 0x00, 0x00, 0x7e,        # F7 continuation
        0x00, 0xff, 0x2f, 0x00,
    ])
    smf = make_smf(480, 0, [track])
    midifile = load(smf)

    assert midifile.tracks[0][0] == Message(
        'sysex', data=(0x43, 0x10, 0x4c), time=0)
    assert midifile.tracks[0][1] == SysexEscapeMessage(
        data=(0x00, 0x00, 0x7e), time=0)

    saved = save(midifile)
    assert b'\xf7\x03\x00\x00\x7e' in saved
    assert b'\xf0\x03\x00\x00\x7e' not in saved
    assert saved == smf


def test_escape_message_rejects_out_of_range_bytes():
    with raises(ValueError):
        SysexEscapeMessage(data=(0x100,))
    with raises(ValueError):
        SysexEscapeMessage(data=(-1,))


def test_escape_message_copy_keeps_type_and_data():
    msg = SysexEscapeMessage(data=(0x90, 60, 64), time=12)
    copy = msg.copy(skip_checks=True, time=0)

    assert isinstance(copy, SysexEscapeMessage)
    assert copy.data == (0x90, 60, 64)
    assert copy.time == 0
    assert copy == SysexEscapeMessage(data=(0x90, 60, 64), time=0)


def test_escape_message_freezes_and_thaws():
    msg = SysexEscapeMessage(data=(0x90, 60, 64), time=5)
    frozen = freeze_message(msg)

    assert isinstance(frozen, FrozenSysexEscapeMessage)
    assert frozen.data == (0x90, 60, 64)
    assert thaw_message(frozen) == msg
