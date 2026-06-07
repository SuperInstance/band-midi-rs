#![forbid(unsafe_code)]

use core::f64::consts::TAU;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MidiMessage {
    NoteOn          { channel: u8, pitch: u8, velocity: u8 },
    NoteOff         { channel: u8, pitch: u8, velocity: u8 },
    PolyPressure    { channel: u8, pitch: u8, pressure: u8 },
    ControlChange   { channel: u8, controller: u8, value: u8 },
    ProgramChange   { channel: u8, program: u8 },
    ChannelPressure { channel: u8, pressure: u8 },
    PitchBend       { channel: u8, value: i16 },
    TimingClock,
    Start,
    Continue,
    Stop,
    ActiveSensing,
    Reset,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MidiError {
    EmptyBuffer,
    UnknownStatus(u8),
    BufferTooShort { needed: usize, got: usize },
}

impl core::fmt::Display for MidiError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::EmptyBuffer => write!(f, "empty buffer"),
            Self::UnknownStatus(s) => write!(f, "unknown status 0x{s:02x}"),
            Self::BufferTooShort { needed, got } => write!(f, "need {needed} bytes, got {got}"),
        }
    }
}

impl MidiMessage {
    /// Parse from raw bytes. Returns (message, bytes_consumed).
    pub fn parse(bytes: &[u8]) -> Result<(Self, usize), MidiError> {
        if bytes.is_empty() { return Err(MidiError::EmptyBuffer); }
        let status = bytes[0];
        let ch = status & 0x0F;
        let need = |n: usize| -> Result<(), MidiError> {
            if bytes.len() < n { Err(MidiError::BufferTooShort { needed: n, got: bytes.len() }) } else { Ok(()) }
        };
        match status & 0xF0 {
            0x80 => { need(3)?; Ok((MidiMessage::NoteOff { channel: ch, pitch: bytes[1], velocity: bytes[2] }, 3)) }
            0x90 => { need(3)?; Ok((MidiMessage::NoteOn  { channel: ch, pitch: bytes[1], velocity: bytes[2] }, 3)) }
            0xA0 => { need(3)?; Ok((MidiMessage::PolyPressure { channel: ch, pitch: bytes[1], pressure: bytes[2] }, 3)) }
            0xB0 => { need(3)?; Ok((MidiMessage::ControlChange { channel: ch, controller: bytes[1], value: bytes[2] }, 3)) }
            0xC0 => { need(2)?; Ok((MidiMessage::ProgramChange { channel: ch, program: bytes[1] }, 2)) }
            0xD0 => { need(2)?; Ok((MidiMessage::ChannelPressure { channel: ch, pressure: bytes[1] }, 2)) }
            0xE0 => {
                need(3)?;
                let raw = ((bytes[2] as i16) << 7) | (bytes[1] as i16);
                Ok((MidiMessage::PitchBend { channel: ch, value: raw - 8192 }, 3))
            }
            0xF0 => match status {
                0xF8 => Ok((MidiMessage::TimingClock, 1)),
                0xFA => Ok((MidiMessage::Start, 1)),
                0xFB => Ok((MidiMessage::Continue, 1)),
                0xFC => Ok((MidiMessage::Stop, 1)),
                0xFE => Ok((MidiMessage::ActiveSensing, 1)),
                0xFF => Ok((MidiMessage::Reset, 1)),
                _ => Err(MidiError::UnknownStatus(status)),
            },
            _ => Err(MidiError::UnknownStatus(status)),
        }
    }

    pub fn to_bytes(&self) -> Vec<u8> {
        match self {
            Self::NoteOn    { channel, pitch, velocity } => vec![0x90 | channel, *pitch, *velocity],
            Self::NoteOff   { channel, pitch, velocity } => vec![0x80 | channel, *pitch, *velocity],
            Self::PolyPressure { channel, pitch, pressure } => vec![0xA0 | channel, *pitch, *pressure],
            Self::ControlChange { channel, controller, value } => vec![0xB0 | channel, *controller, *value],
            Self::ProgramChange { channel, program } => vec![0xC0 | channel, *program],
            Self::ChannelPressure { channel, pressure } => vec![0xD0 | channel, *pressure],
            Self::PitchBend { channel, value } => {
                let raw = (*value + 8192) as u16;
                vec![0xE0 | channel, (raw & 0x7F) as u8, ((raw >> 7) & 0x7F) as u8]
            }
            Self::TimingClock  => vec![0xF8],
            Self::Start        => vec![0xFA],
            Self::Continue     => vec![0xFB],
            Self::Stop         => vec![0xFC],
            Self::ActiveSensing => vec![0xFE],
            Self::Reset        => vec![0xFF],
        }
    }
}

/// 3D tensor coordinate: pitch [0,1], velocity [0,1], phase [0, 2π).
#[derive(Debug, Clone, PartialEq)]
pub struct TensorCoord {
    pub pitch_norm: f64,
    pub vel_norm: f64,
    pub phase: f64,
}

impl TensorCoord {
    pub fn new(pitch_norm: f64, vel_norm: f64, phase: f64) -> Self {
        TensorCoord { pitch_norm, vel_norm, phase }
    }

    pub fn distance(&self, other: &Self) -> f64 {
        let dp = self.pitch_norm - other.pitch_norm;
        let dv = self.vel_norm   - other.vel_norm;
        let dph = (self.phase - other.phase) / TAU;
        (dp * dp + dv * dv + dph * dph).sqrt()
    }
}

pub fn midi_to_tensor(msg: &MidiMessage, time_phase: f64) -> Option<TensorCoord> {
    let phase = time_phase % TAU;
    match msg {
        MidiMessage::NoteOn  { pitch, velocity, .. } |
        MidiMessage::NoteOff { pitch, velocity, .. } =>
            Some(TensorCoord::new(*pitch as f64 / 127.0, *velocity as f64 / 127.0, phase)),
        MidiMessage::ControlChange { controller, value, .. } =>
            Some(TensorCoord::new(*controller as f64 / 127.0, *value as f64 / 127.0, phase)),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::f64::consts::PI;

    fn rt(msg: &MidiMessage) -> MidiMessage { MidiMessage::parse(&msg.to_bytes()).unwrap().0 }

    #[test] fn note_on_round_trip() {
        let m = MidiMessage::NoteOn { channel: 0, pitch: 60, velocity: 100 };
        assert_eq!(rt(&m), m);
    }
    #[test] fn note_off_round_trip() {
        let m = MidiMessage::NoteOff { channel: 1, pitch: 64, velocity: 0 };
        assert_eq!(rt(&m), m);
    }
    #[test] fn control_change_round_trip() {
        let m = MidiMessage::ControlChange { channel: 2, controller: 7, value: 100 };
        assert_eq!(rt(&m), m);
    }
    #[test] fn program_change_round_trip() {
        let m = MidiMessage::ProgramChange { channel: 0, program: 25 };
        assert_eq!(rt(&m), m);
    }
    #[test] fn pitch_bend_zero() { assert_eq!(rt(&MidiMessage::PitchBend { channel: 0, value: 0 }), MidiMessage::PitchBend { channel: 0, value: 0 }); }
    #[test] fn pitch_bend_max_pos() { assert_eq!(rt(&MidiMessage::PitchBend { channel: 0, value: 8191 }), MidiMessage::PitchBend { channel: 0, value: 8191 }); }
    #[test] fn pitch_bend_max_neg() { assert_eq!(rt(&MidiMessage::PitchBend { channel: 0, value: -8192 }), MidiMessage::PitchBend { channel: 0, value: -8192 }); }
    #[test] fn timing_clock() { assert_eq!(rt(&MidiMessage::TimingClock), MidiMessage::TimingClock); }
    #[test] fn start_stop_continue() {
        assert_eq!(rt(&MidiMessage::Start), MidiMessage::Start);
        assert_eq!(rt(&MidiMessage::Stop), MidiMessage::Stop);
        assert_eq!(rt(&MidiMessage::Continue), MidiMessage::Continue);
    }
    #[test] fn reset_round_trip() { assert_eq!(rt(&MidiMessage::Reset), MidiMessage::Reset); }
    #[test] fn poly_pressure_round_trip() {
        let m = MidiMessage::PolyPressure { channel: 3, pitch: 50, pressure: 64 };
        assert_eq!(rt(&m), m);
    }
    #[test] fn channel_pressure_round_trip() {
        let m = MidiMessage::ChannelPressure { channel: 0, pressure: 100 };
        assert_eq!(rt(&m), m);
    }
    #[test] fn parse_empty_err() { assert!(matches!(MidiMessage::parse(&[]), Err(MidiError::EmptyBuffer))); }
    #[test] fn parse_too_short_err() { assert!(matches!(MidiMessage::parse(&[0x90]), Err(MidiError::BufferTooShort { .. }))); }
    #[test] fn parse_unknown_status_err() { assert!(matches!(MidiMessage::parse(&[0xF1]), Err(MidiError::UnknownStatus(_)))); }
    #[test] fn error_display_empty() { assert!(!MidiError::EmptyBuffer.to_string().is_empty()); }
    #[test] fn note_on_tensor_full() {
        let m = MidiMessage::NoteOn { channel: 0, pitch: 127, velocity: 127 };
        let t = midi_to_tensor(&m, 0.0).unwrap();
        assert!((t.pitch_norm - 1.0).abs() < 1e-9);
        assert!((t.vel_norm   - 1.0).abs() < 1e-9);
    }
    #[test] fn note_on_tensor_zero() {
        let m = MidiMessage::NoteOn { channel: 0, pitch: 0, velocity: 0 };
        let t = midi_to_tensor(&m, PI).unwrap();
        assert!(t.pitch_norm.abs() < 1e-9);
        assert!((t.phase - PI).abs() < 1e-9);
    }
    #[test] fn tensor_distance_self_zero() {
        let t = TensorCoord::new(0.5, 0.5, PI);
        assert!(t.distance(&t) < 1e-9);
    }
    #[test] fn tensor_distance_symmetry() {
        let a = TensorCoord::new(0.0, 0.0, 0.0);
        let b = TensorCoord::new(1.0, 1.0, PI);
        assert!((a.distance(&b) - b.distance(&a)).abs() < 1e-9);
    }
    #[test] fn system_messages_no_tensor() {
        assert!(midi_to_tensor(&MidiMessage::Start, 0.0).is_none());
        assert!(midi_to_tensor(&MidiMessage::Stop, 0.0).is_none());
        assert!(midi_to_tensor(&MidiMessage::Reset, 0.0).is_none());
    }
    #[test] fn note_on_all_channels() {
        for ch in 0u8..16 {
            let m = MidiMessage::NoteOn { channel: ch, pitch: 60, velocity: 80 };
            assert_eq!(rt(&m), m);
        }
    }
    #[test] fn active_sensing_round_trip() { assert_eq!(rt(&MidiMessage::ActiveSensing), MidiMessage::ActiveSensing); }
    #[test] fn control_change_tensor() {
        let m = MidiMessage::ControlChange { channel: 0, controller: 64, value: 64 };
        let t = midi_to_tensor(&m, 0.0).unwrap();
        assert!((t.pitch_norm - 64.0/127.0).abs() < 1e-9);
    }
}
