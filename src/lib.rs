//! MIDI I/O, tensor mapping, and groove timing for autonomous band agents.
//!
//! This crate provides:
//! - [`note`]: MIDI note structures and encoding/decoding
//! - [`tensor_map`]: Tensor representations of MIDI events for ML pipelines
//! - [`groove`]: Swing, humanization, and quantization engine
//! - [`clock_sync`]: MIDI clock synchronization state machine
//! - [`io`]: Software MIDI buffer for event scheduling and encoding

// ─────────────────────────────── note ───────────────────────────────────────

/// MIDI note structures and event types.
pub mod note {
    /// A single MIDI note with pitch, velocity, duration, and channel.
    #[derive(Debug, Clone, PartialEq)]
    pub struct Note {
        /// MIDI pitch value (0–127).
        pub pitch: u8,
        /// MIDI velocity value (0–127). A value of 0 indicates silence.
        pub velocity: u8,
        /// Note duration in seconds.
        pub duration: f64,
        /// MIDI channel (0–15).
        pub channel: u8,
    }

    /// A MIDI Note On event.
    #[derive(Debug, Clone, PartialEq)]
    pub struct NoteOn {
        /// MIDI pitch value (0–127).
        pub pitch: u8,
        /// MIDI velocity value (0–127).
        pub velocity: u8,
        /// MIDI channel (0–15).
        pub channel: u8,
        /// Event timestamp in seconds.
        pub timestamp: f64,
    }

    /// A MIDI Note Off event.
    #[derive(Debug, Clone, PartialEq)]
    pub struct NoteOff {
        /// MIDI pitch value (0–127).
        pub pitch: u8,
        /// MIDI channel (0–15).
        pub channel: u8,
        /// Event timestamp in seconds.
        pub timestamp: f64,
    }

    /// A MIDI event discriminant covering the most common message types.
    #[derive(Debug, Clone, PartialEq)]
    pub enum MidiEvent {
        /// A note-on message.
        NoteOn(NoteOn),
        /// A note-off message.
        NoteOff(NoteOff),
        /// A MIDI timing clock pulse (24 per quarter note).
        Clock,
        /// A MIDI Start message.
        Start,
        /// A MIDI Stop message.
        Stop,
        /// A MIDI Continue message.
        Continue,
    }

    impl Note {
        /// Create a new [`Note`].
        pub fn new(pitch: u8, velocity: u8, duration: f64, channel: u8) -> Self {
            Self { pitch, velocity, duration, channel }
        }

        /// Return the pitch class (0–11) by computing `pitch % 12`.
        pub fn pitch_class(&self) -> u8 {
            self.pitch % 12
        }

        /// Return the octave number by computing `pitch / 12`.
        pub fn octave(&self) -> u8 {
            self.pitch / 12
        }

        /// Return `true` if velocity is 0 (silent note).
        pub fn is_silence(&self) -> bool {
            self.velocity == 0
        }

        /// Encode the note as a 3-byte MIDI Note On message.
        ///
        /// Returns `[status, pitch, velocity]` where status = `0x90 | channel`.
        pub fn encode(&self) -> [u8; 3] {
            [0x90 | (self.channel & 0x0F), self.pitch, self.velocity]
        }

        /// Attempt to decode a 3-byte MIDI Note On message into a [`Note`].
        ///
        /// Returns `None` if the status byte is not a Note On message (`0x90`–`0x9F`).
        pub fn decode(bytes: [u8; 3], duration: f64) -> Option<Self> {
            let status = bytes[0];
            if status & 0xF0 == 0x90 {
                let channel = status & 0x0F;
                Some(Self::new(bytes[1], bytes[2], duration, channel))
            } else {
                None
            }
        }
    }
}

// ─────────────────────────────── tensor_map ──────────────────────────────────

/// Tensor representations of MIDI events for ML/DSP pipelines.
pub mod tensor_map {
    use crate::note::Note;

    /// A single tensor-encoded MIDI event.
    #[derive(Debug, Clone, PartialEq)]
    pub struct TensorEvent {
        /// Pitch class (0–11), derived from the note's MIDI pitch.
        pub dimension: u8,
        /// Normalised velocity in `[0, 1]` (velocity / 127.0).
        pub weight: f64,
        /// Rhythm pattern encoded as 4 floats (one per 16th note in a beat).
        pub kernel: [f64; 4],
    }

    /// A collection of simultaneous [`TensorEvent`]s representing a chord or
    /// cluster of notes at a single point in time.
    #[derive(Debug, Clone, PartialEq)]
    pub struct TensorSlice {
        /// All events that occur simultaneously.
        pub events: Vec<TensorEvent>,
    }

    /// An ordered sequence of timestamped [`TensorSlice`]s.
    #[derive(Debug, Clone, PartialEq)]
    pub struct TensorSequence {
        /// `(timestamp_seconds, slice)` pairs in chronological order.
        pub slices: Vec<(f64, TensorSlice)>,
    }

    impl TensorEvent {
        /// Build a [`TensorEvent`] from a [`Note`] and a 4-element rhythm pattern.
        pub fn from_note(note: &Note, rhythm_pattern: [f64; 4]) -> Self {
            Self {
                dimension: note.pitch_class(),
                weight: note.velocity as f64 / 127.0,
                kernel: rhythm_pattern,
            }
        }

        /// Compute the dot product of two [`TensorEvent`]s.
        ///
        /// Combines weight product with the inner product of their kernels.
        pub fn dot_product(&self, other: &TensorEvent) -> f64 {
            let kernel_dot: f64 = self
                .kernel
                .iter()
                .zip(other.kernel.iter())
                .map(|(a, b)| a * b)
                .sum();
            self.weight * other.weight + kernel_dot
        }
    }

    impl TensorSlice {
        /// Build a [`TensorSlice`] from a slice of notes and a shared rhythm pattern.
        pub fn from_notes(notes: &[Note], rhythm_pattern: [f64; 4]) -> Self {
            Self {
                events: notes
                    .iter()
                    .map(|n| TensorEvent::from_note(n, rhythm_pattern))
                    .collect(),
            }
        }

        /// Sum of all event weights in this slice.
        pub fn activation_sum(&self) -> f64 {
            self.events.iter().map(|e| e.weight).sum()
        }
    }

    impl TensorSequence {
        /// Create an empty [`TensorSequence`].
        pub fn new() -> Self {
            Self { slices: Vec::new() }
        }

        /// Append a slice at the given timestamp.
        pub fn push(&mut self, timestamp: f64, slice: TensorSlice) {
            self.slices.push((timestamp, slice));
        }

        /// Sum of all event weights across all slices.
        pub fn total_weight(&self) -> f64 {
            self.slices
                .iter()
                .flat_map(|(_, s)| s.events.iter())
                .map(|e| e.weight)
                .sum()
        }
    }

    impl Default for TensorSequence {
        fn default() -> Self {
            Self::new()
        }
    }
}

// ─────────────────────────────── groove ──────────────────────────────────────

/// Swing, humanization, and beat-grid quantization engine.
pub mod groove {
    /// Applies swing feel, subtle timing variations, and quantization to a
    /// beat grid.
    #[derive(Debug, Clone, PartialEq)]
    pub struct GrooveEngine {
        /// Swing amount in `[0, 0.5]`. `0` = straight, `0.5` = full triplet feel.
        pub swing: f64,
        /// Maximum random humanization offset in milliseconds.
        pub humanize: f64,
        /// Tempo in beats per minute.
        pub tempo_bpm: f64,
    }

    impl GrooveEngine {
        /// Create a new [`GrooveEngine`] with the given tempo and sensible
        /// defaults: `swing = 0.1`, `humanize = 5.0 ms`.
        pub fn new(tempo_bpm: f64) -> Self {
            Self { swing: 0.1, humanize: 5.0, tempo_bpm }
        }

        /// Duration of a single beat in milliseconds at the current tempo.
        pub fn beat_duration_ms(&self) -> f64 {
            60_000.0 / self.tempo_bpm
        }

        /// Compute the timing offset (in ms) for `beat_index` using swing and
        /// deterministic humanization derived from `seed`.
        ///
        /// - Even-indexed beats are pushed *forward* by `swing × beat_duration_ms`.
        /// - Odd-indexed beats are pushed *backward* by the same amount.
        /// - A deterministic humanize jitter is added: `(seed % 1000) / 1000 * humanize - humanize/2`.
        pub fn beat_offset(&self, beat_index: u64, seed: u64) -> f64 {
            let beat_ms = self.beat_duration_ms();
            let swing_offset = if beat_index.is_multiple_of(2) {
                self.swing * beat_ms
            } else {
                -(self.swing * beat_ms)
            };
            let human_offset =
                (seed % 1000) as f64 / 1000.0 * self.humanize - self.humanize / 2.0;
            swing_offset + human_offset
        }

        /// Snap `timestamp_ms` to the nearest beat grid position.
        pub fn quantize(&self, timestamp_ms: f64) -> f64 {
            let beat_ms = self.beat_duration_ms();
            (timestamp_ms / beat_ms).round() * beat_ms
        }
    }
}

// ─────────────────────────────── clock_sync ──────────────────────────────────

/// MIDI clock synchronization state machine.
///
/// MIDI clock sends 24 ticks per quarter note; this module tracks those ticks
/// to derive the external tempo.
pub mod clock_sync {
    /// Tracks incoming MIDI clock ticks and derives BPM from tick intervals.
    #[derive(Debug, Clone, PartialEq)]
    pub struct ClockSync {
        /// `true` when a MIDI Start message has been received.
        pub is_synced: bool,
        /// Derived external BPM once 24 ticks have been received.
        pub external_bpm: Option<f64>,
        /// Running tick counter (resets every 24 ticks).
        pub tick_count: u64,
        /// Timestamp of the first tick in the current 24-tick window.
        pub last_tick_time: f64,
    }

    impl ClockSync {
        /// Create a new, unsynced [`ClockSync`] instance.
        pub fn new() -> Self {
            Self {
                is_synced: false,
                external_bpm: None,
                tick_count: 0,
                last_tick_time: 0.0,
            }
        }

        /// Process a single MIDI clock tick arriving at `time` (seconds).
        ///
        /// Every 24 ticks a new BPM estimate is computed and the window resets.
        pub fn on_clock_tick(&mut self, time: f64) {
            if self.tick_count == 0 {
                self.last_tick_time = time;
            }
            self.tick_count += 1;
            if self.tick_count >= 24 {
                let elapsed = time - self.last_tick_time;
                if elapsed > 0.0 {
                    self.external_bpm = Some(60.0 * 24.0 / elapsed);
                }
                self.tick_count = 0;
                self.last_tick_time = time;
            }
        }

        /// Handle a MIDI Start message: mark as synced and reset the tick counter.
        pub fn on_start(&mut self) {
            self.is_synced = true;
            self.tick_count = 0;
        }

        /// Handle a MIDI Stop message: mark as unsynced.
        pub fn on_stop(&mut self) {
            self.is_synced = false;
        }

        /// Return the current external BPM if the clock is synced.
        pub fn current_bpm(&self) -> Option<f64> {
            if self.is_synced { self.external_bpm } else { None }
        }
    }

    impl Default for ClockSync {
        fn default() -> Self {
            Self::new()
        }
    }
}

// ─────────────────────────────── io ──────────────────────────────────────────

/// Software MIDI event buffer for scheduling and encoding events.
///
/// No hardware I/O is performed; this module provides an in-memory store
/// suitable for offline rendering, testing, and agent communication.
pub mod io {
    use crate::note::MidiEvent;

    /// An in-memory buffer of timestamped [`MidiEvent`]s.
    #[derive(Debug, Clone, PartialEq)]
    pub struct MidiBuffer {
        /// `(timestamp_seconds, event)` pairs.
        pub events: Vec<(f64, MidiEvent)>,
    }

    impl MidiBuffer {
        /// Create an empty [`MidiBuffer`].
        pub fn new() -> Self {
            Self { events: Vec::new() }
        }

        /// Append an event at `timestamp` (seconds).
        pub fn push(&mut self, timestamp: f64, event: MidiEvent) {
            self.events.push((timestamp, event));
        }

        /// Return references to all events whose timestamp falls in `[start, end)`.
        pub fn events_in_range(&self, start: f64, end: f64) -> Vec<&MidiEvent> {
            self.events
                .iter()
                .filter(|(t, _)| *t >= start && *t < end)
                .map(|(_, e)| e)
                .collect()
        }

        /// Encode all `NoteOn` events in the buffer to raw MIDI bytes (3 bytes each).
        ///
        /// Other event types are skipped.
        pub fn encode_buffer(&self) -> Vec<u8> {
            let mut out = Vec::new();
            for (_, event) in &self.events {
                if let MidiEvent::NoteOn(on) = event {
                    out.push(0x90 | (on.channel & 0x0F));
                    out.push(on.pitch);
                    out.push(on.velocity);
                }
            }
            out
        }

        /// Remove all events from the buffer.
        pub fn clear(&mut self) {
            self.events.clear();
        }
    }

    impl Default for MidiBuffer {
        fn default() -> Self {
            Self::new()
        }
    }
}

// ─────────────────────────────── tests ───────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::clock_sync::ClockSync;
    use super::groove::GrooveEngine;
    use super::io::MidiBuffer;
    use super::note::{MidiEvent, Note, NoteOff, NoteOn};
    use super::tensor_map::{TensorEvent, TensorSequence, TensorSlice};

    // ── note ────────────────────────────────────────────────────────────────

    #[test]
    fn note_new() {
        let n = Note::new(60, 100, 1.0, 0);
        assert_eq!(n.pitch, 60);
        assert_eq!(n.velocity, 100);
        assert_eq!(n.duration, 1.0);
        assert_eq!(n.channel, 0);
    }

    #[test]
    fn note_pitch_class_c() {
        // Middle C (MIDI 60) has pitch class 0
        let n = Note::new(60, 80, 0.5, 0);
        assert_eq!(n.pitch_class(), 0);
    }

    #[test]
    fn note_pitch_class_a() {
        // MIDI 57 = A3, pitch class 9
        let n = Note::new(57, 80, 0.5, 0);
        assert_eq!(n.pitch_class(), 9);
    }

    #[test]
    fn note_octave_middle_c() {
        // MIDI 60 / 12 = 5
        let n = Note::new(60, 80, 0.5, 0);
        assert_eq!(n.octave(), 5);
    }

    #[test]
    fn note_is_silence_true() {
        let n = Note::new(60, 0, 0.5, 0);
        assert!(n.is_silence());
    }

    #[test]
    fn note_is_silence_false() {
        let n = Note::new(60, 64, 0.5, 0);
        assert!(!n.is_silence());
    }

    #[test]
    fn note_encode() {
        let n = Note::new(60, 100, 1.0, 0);
        assert_eq!(n.encode(), [0x90, 60, 100]);
    }

    #[test]
    fn note_encode_channel() {
        let n = Note::new(69, 90, 1.0, 3);
        let enc = n.encode();
        assert_eq!(enc[0], 0x93); // 0x90 | 3
        assert_eq!(enc[1], 69);
        assert_eq!(enc[2], 90);
    }

    #[test]
    fn note_decode_roundtrip() {
        let n = Note::new(64, 80, 2.0, 1);
        let bytes = n.encode();
        let decoded = Note::decode(bytes, 2.0).expect("decode should succeed");
        assert_eq!(decoded, n);
    }

    #[test]
    fn note_decode_invalid_status() {
        // 0x80 is Note Off, not Note On
        let result = Note::decode([0x80, 60, 0], 1.0);
        assert!(result.is_none());
    }

    #[test]
    fn midi_event_variants_constructable() {
        let on = MidiEvent::NoteOn(NoteOn { pitch: 60, velocity: 100, channel: 0, timestamp: 0.0 });
        let off = MidiEvent::NoteOff(NoteOff { pitch: 60, channel: 0, timestamp: 1.0 });
        let clock = MidiEvent::Clock;
        let start = MidiEvent::Start;
        let stop = MidiEvent::Stop;
        let cont = MidiEvent::Continue;
        // just assert they are distinguishable
        assert_ne!(on, off);
        assert_eq!(clock, MidiEvent::Clock);
        assert_eq!(start, MidiEvent::Start);
        assert_eq!(stop, MidiEvent::Stop);
        assert_eq!(cont, MidiEvent::Continue);
    }

    // ── tensor_map ──────────────────────────────────────────────────────────

    #[test]
    fn tensor_event_from_note_weight() {
        let n = Note::new(60, 127, 1.0, 0);
        let te = TensorEvent::from_note(&n, [0.25; 4]);
        assert!((te.weight - 1.0).abs() < 1e-9);
    }

    #[test]
    fn tensor_event_from_note_dimension() {
        // pitch 63 = pitch class 3
        let n = Note::new(63, 64, 1.0, 0);
        let te = TensorEvent::from_note(&n, [0.0; 4]);
        assert_eq!(te.dimension, 3);
    }

    #[test]
    fn tensor_event_dot_product() {
        let n = Note::new(60, 127, 1.0, 0);
        let a = TensorEvent::from_note(&n, [1.0, 0.0, 0.0, 0.0]);
        let b = TensorEvent::from_note(&n, [1.0, 0.0, 0.0, 0.0]);
        // weight*weight + 1*1 + 0+0+0 = 1.0 + 1.0 = 2.0
        let dp = a.dot_product(&b);
        assert!((dp - 2.0).abs() < 1e-9);
    }

    #[test]
    fn tensor_slice_from_notes() {
        let notes = vec![Note::new(60, 64, 1.0, 0), Note::new(64, 64, 1.0, 0)];
        let ts = TensorSlice::from_notes(&notes, [0.5; 4]);
        assert_eq!(ts.events.len(), 2);
    }

    #[test]
    fn tensor_slice_activation_sum() {
        let notes = vec![Note::new(60, 127, 1.0, 0), Note::new(64, 127, 1.0, 0)];
        let ts = TensorSlice::from_notes(&notes, [0.0; 4]);
        let sum = ts.activation_sum();
        assert!((sum - 2.0).abs() < 1e-9);
    }

    #[test]
    fn tensor_sequence_push_and_total_weight() {
        let mut seq = TensorSequence::new();
        let notes = vec![Note::new(60, 127, 1.0, 0)];
        let sl = TensorSlice::from_notes(&notes, [0.0; 4]);
        seq.push(0.0, sl.clone());
        seq.push(0.5, sl);
        // Two slices each with weight 1.0
        assert!((seq.total_weight() - 2.0).abs() < 1e-9);
    }

    // ── groove ──────────────────────────────────────────────────────────────

    #[test]
    fn groove_beat_duration_120bpm() {
        let g = GrooveEngine::new(120.0);
        assert!((g.beat_duration_ms() - 500.0).abs() < 1e-9);
    }

    #[test]
    fn groove_beat_offset_swing_sign_alternates() {
        let g = GrooveEngine::new(120.0);
        // Use seed 0 to eliminate humanize randomness contribution
        let even = g.beat_offset(0, 0);
        let odd = g.beat_offset(1, 0);
        assert!(even > 0.0, "even beat offset should be positive");
        assert!(odd < 0.0, "odd beat offset should be negative");
    }

    #[test]
    fn groove_quantize_snaps() {
        let g = GrooveEngine::new(120.0); // 500 ms per beat
        // 750 ms is between beat 1 (500ms) and beat 2 (1000ms) — closer to beat 2
        let q = g.quantize(750.0);
        assert!((q - 1000.0).abs() < 1e-9);
    }

    #[test]
    fn groove_quantize_snaps_to_zero() {
        let g = GrooveEngine::new(120.0);
        let q = g.quantize(200.0); // closer to beat 0 (0ms)
        assert!((q - 0.0).abs() < 1e-9);
    }

    // ── clock_sync ──────────────────────────────────────────────────────────

    #[test]
    fn clock_sync_new_unsynced() {
        let cs = ClockSync::new();
        assert!(!cs.is_synced);
        assert!(cs.external_bpm.is_none());
    }

    #[test]
    fn clock_sync_on_start_sets_synced() {
        let mut cs = ClockSync::new();
        cs.on_start();
        assert!(cs.is_synced);
    }

    #[test]
    fn clock_sync_on_stop_unsynced() {
        let mut cs = ClockSync::new();
        cs.on_start();
        cs.on_stop();
        assert!(!cs.is_synced);
    }

    #[test]
    fn clock_sync_on_clock_tick_accumulates() {
        let mut cs = ClockSync::new();
        cs.on_start();
        // Feed 24 ticks 1ms apart (BPM should be 60*24/0.023 ≈ 2500)
        for i in 0..24u64 {
            cs.on_clock_tick(i as f64 * 0.001);
        }
        // After 24 ticks BPM should be set
        assert!(cs.external_bpm.is_some());
    }

    #[test]
    fn clock_sync_current_bpm_requires_synced() {
        let mut cs = ClockSync::new();
        // Feed ticks without starting
        for i in 0..24u64 {
            cs.on_clock_tick(i as f64 * 0.001);
        }
        // Even with bpm set, current_bpm returns None when not synced
        cs.on_stop();
        assert!(cs.current_bpm().is_none());
    }

    // ── io ──────────────────────────────────────────────────────────────────

    #[test]
    fn midi_buffer_push_and_len() {
        let mut buf = MidiBuffer::new();
        let ev = MidiEvent::NoteOn(NoteOn { pitch: 60, velocity: 100, channel: 0, timestamp: 0.0 });
        buf.push(0.0, ev);
        assert_eq!(buf.events.len(), 1);
    }

    #[test]
    fn midi_buffer_events_in_range() {
        let mut buf = MidiBuffer::new();
        let make_on = |t: f64| {
            MidiEvent::NoteOn(NoteOn { pitch: 60, velocity: 100, channel: 0, timestamp: t })
        };
        buf.push(0.0, make_on(0.0));
        buf.push(0.5, make_on(0.5));
        buf.push(1.0, make_on(1.0));
        let range = buf.events_in_range(0.0, 1.0);
        assert_eq!(range.len(), 2); // [0.0, 1.0) excludes t=1.0
    }

    #[test]
    fn midi_buffer_encode_buffer() {
        let mut buf = MidiBuffer::new();
        buf.push(
            0.0,
            MidiEvent::NoteOn(NoteOn { pitch: 60, velocity: 100, channel: 0, timestamp: 0.0 }),
        );
        buf.push(0.1, MidiEvent::Clock); // should be skipped
        let bytes = buf.encode_buffer();
        assert_eq!(bytes, vec![0x90, 60, 100]);
    }

    #[test]
    fn midi_buffer_clear() {
        let mut buf = MidiBuffer::new();
        buf.push(
            0.0,
            MidiEvent::NoteOn(NoteOn { pitch: 60, velocity: 100, channel: 0, timestamp: 0.0 }),
        );
        buf.clear();
        assert!(buf.events.is_empty());
    }
}
