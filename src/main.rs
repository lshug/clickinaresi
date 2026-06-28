//! clickinaresi — audio player for live performance.
//!
//! The window shows the currently-playing track in large text at the top, a
//! transport bar (play/pause, stop, next, previous) and a scrollable list of
//! tracks below. The four transport buttons are MIDI-mappable from the
//! configuration window via "MIDI learn".

use std::fs::File;
use std::io::BufReader;
use std::path::{Path, PathBuf};
use std::sync::mpsc::{Receiver, Sender, channel};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use eframe::{Storage, egui};
use egui::RichText;
use midir::{MidiInput, MidiInputConnection};
use rodio::{Decoder, OutputStream, OutputStreamHandle, Sink};

const AUDIO_EXTENSIONS: &[&str] = &["mp3", "wav", "flac", "ogg"];
/// Storage keys under which state is persisted across launches.
const LAST_PLAYLIST_KEY: &str = "last_playlist";
const MIDI_BINDINGS_KEY: &str = "midi_bindings";
const MIDI_PORT_KEY: &str = "midi_port";
const VOLUME_KEY: &str = "volume_db";

fn main() -> eframe::Result {
    let native_options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([900.0, 700.0])
            .with_min_inner_size([480.0, 360.0])
            .with_maximized(true)
            .with_title("clickinaresi"),
        // Don't restore the saved window geometry — it would override the
        // maximized request above. App state (playlist, MIDI) is still
        // persisted via `App::save`.
        persist_window: false,
        ..Default::default()
    };
    eframe::run_native(
        "clickinaresi",
        native_options,
        Box::new(|cc| Ok(Box::new(App::new(cc)?))),
    )
}

// ---------------------------------------------------------------------------
// Transport actions
// ---------------------------------------------------------------------------

/// The four transport actions. These are the only things a MIDI message can
/// trigger, and they are the unit of communication between the MIDI thread and
/// the UI thread.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Action {
    PlayPause,
    Stop,
    Next,
    Prev,
}

impl Action {
    const ALL: [Action; 4] = [Action::PlayPause, Action::Stop, Action::Next, Action::Prev];

    fn label(self) -> &'static str {
        match self {
            Action::PlayPause => "Play / Pause",
            Action::Stop => "Stop",
            Action::Next => "Next",
            Action::Prev => "Previous",
        }
    }

    /// Stable identifier used when persisting bindings.
    fn key(self) -> &'static str {
        match self {
            Action::PlayPause => "play_pause",
            Action::Stop => "stop",
            Action::Next => "next",
            Action::Prev => "prev",
        }
    }

    fn from_key(s: &str) -> Option<Action> {
        Action::ALL.into_iter().find(|a| a.key() == s)
    }
}

// ---------------------------------------------------------------------------
// Audio player
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq, Eq)]
enum PlaybackState {
    Stopped,
    Playing,
    Paused,
}

/// Wraps rodio. The `OutputStream` must be kept alive for audio to play, hence
/// the field is held even though it is never read directly.
struct Player {
    _stream: OutputStream,
    handle: OutputStreamHandle,
    sink: Sink,
    state: PlaybackState,
    /// Display name of the track currently loaded into the sink.
    current: Option<String>,
    /// Output gain in decibels (0 dB = unity). Applied to every sink.
    volume_db: f32,
}

impl Player {
    fn new() -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        let (stream, handle) = OutputStream::try_default()?;
        let sink = Sink::try_new(&handle)?;
        Ok(Self {
            _stream: stream,
            handle,
            sink,
            state: PlaybackState::Stopped,
            current: None,
            volume_db: 0.0,
        })
    }

    /// Linear amplitude factor for the current gain (rodio's `set_volume` unit).
    fn amplitude(&self) -> f32 {
        10f32.powf(self.volume_db / 20.0)
    }

    fn set_volume_db(&mut self, db: f32) {
        self.volume_db = db;
        self.sink.set_volume(self.amplitude());
    }

    /// Loads `path` into a fresh sink and starts playback immediately.
    fn play_path(&mut self, path: &Path, name: String) -> Result<(), Box<dyn std::error::Error>> {
        let file = File::open(path)?;
        let source = Decoder::new(BufReader::new(file))?;
        // A stopped sink cannot be reused, so always start from a clean one.
        self.sink = Sink::try_new(&self.handle)?;
        self.sink.set_volume(self.amplitude());
        self.sink.append(source);
        self.sink.play();
        self.state = PlaybackState::Playing;
        self.current = Some(name);
        Ok(())
    }

    fn pause(&mut self) {
        self.sink.pause();
        self.state = PlaybackState::Paused;
    }

    fn resume(&mut self) {
        self.sink.play();
        self.state = PlaybackState::Playing;
    }

    fn stop(&mut self) {
        self.sink.stop();
        self.state = PlaybackState::Stopped;
        self.current = None;
    }

    /// Detects natural end-of-track so the UI reflects "stopped" once the sink
    /// drains. Call once per frame.
    fn poll(&mut self) {
        if self.state == PlaybackState::Playing && self.sink.empty() {
            self.state = PlaybackState::Stopped;
            self.current = None;
        }
    }
}

// ---------------------------------------------------------------------------
// MIDI
// ---------------------------------------------------------------------------

/// A MIDI binding is matched on the first two bytes (status + note/CC number),
/// ignoring velocity so the same physical key always maps to one action.
type MidiKey = [u8; 2];

/// Shared between the MIDI input thread (callback) and the UI thread.
#[derive(Default)]
struct MidiBindings {
    /// When set, the next incoming trigger is captured as this action's binding
    /// ("MIDI learn") instead of being dispatched.
    learning: Option<Action>,
    map: Vec<(MidiKey, Action)>,
}

impl MidiBindings {
    fn binding_for(&self, action: Action) -> Option<MidiKey> {
        self.map.iter().find(|(_, a)| *a == action).map(|(k, _)| *k)
    }

    fn clear(&mut self, action: Action) {
        self.map.retain(|(_, a)| *a != action);
    }
}

/// Handles a raw MIDI message on the input thread. Note-on (0x9n) with
/// non-zero velocity and control-change (0xBn) with non-zero value count as
/// "triggers"; everything else (including note-off / note-on velocity 0) is
/// ignored so a key press fires exactly once.
fn handle_midi(
    msg: &[u8],
    bindings: &Arc<Mutex<MidiBindings>>,
    tx: &Sender<Action>,
    ctx: &egui::Context,
) {
    if msg.len() < 3 {
        return;
    }
    let status = msg[0] & 0xF0;
    let is_trigger = matches!(status, 0x90 | 0xB0) && msg[2] > 0;
    if !is_trigger {
        return;
    }
    let key: MidiKey = [msg[0], msg[1]];

    let mut b = bindings.lock().unwrap();
    if let Some(action) = b.learning.take() {
        // Capture the binding: a key maps to one action, an action to one key.
        b.map.retain(|(k, a)| *a != action && *k != key);
        b.map.push((key, action));
        ctx.request_repaint();
    } else if let Some((_, action)) = b.map.iter().find(|(k, _)| *k == key) {
        let _ = tx.send(*action);
        ctx.request_repaint();
    }
}

/// Serializes bindings to a JSON array of `{status, data1, action}` objects.
fn bindings_to_json(map: &[(MidiKey, Action)]) -> String {
    let items: Vec<serde_json::Value> = map
        .iter()
        .map(|(k, a)| {
            serde_json::json!({ "status": k[0], "data1": k[1], "action": a.key() })
        })
        .collect();
    serde_json::Value::Array(items).to_string()
}

fn bindings_from_json(text: &str) -> Vec<(MidiKey, Action)> {
    let mut out = Vec::new();
    let Ok(serde_json::Value::Array(items)) = serde_json::from_str::<serde_json::Value>(text)
    else {
        return out;
    };
    for item in items {
        if let (Some(status), Some(data1), Some(action)) = (
            item["status"].as_u64(),
            item["data1"].as_u64(),
            item["action"].as_str().and_then(Action::from_key),
        ) {
            out.push(([status as u8, data1 as u8], action));
        }
    }
    out
}

fn fmt_key(key: MidiKey) -> String {
    let kind = match key[0] & 0xF0 {
        0x90 => "Note",
        0xB0 => "CC",
        _ => "?",
    };
    let channel = (key[0] & 0x0F) + 1;
    format!("{kind} {} (ch {channel})", key[1])
}

// ---------------------------------------------------------------------------
// Track list
// ---------------------------------------------------------------------------

struct Track {
    path: PathBuf,
    name: String,
}

/// Returns at most the first `n` characters of `s` (char-aware, never splits a
/// multi-byte character).
fn first_chars(s: &str, n: usize) -> String {
    s.chars().take(n).collect()
}

fn is_audio_file(path: &Path) -> bool {
    path.extension()
        .and_then(|e| e.to_str())
        .map(|e| AUDIO_EXTENSIONS.contains(&e.to_lowercase().as_str()))
        .unwrap_or(false)
}

fn scan_dir(dir: &Path, out: &mut Vec<PathBuf>) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    let mut paths: Vec<PathBuf> = entries.flatten().map(|e| e.path()).collect();
    paths.sort();
    for path in paths {
        if path.is_dir() {
            scan_dir(&path, out);
        } else if is_audio_file(&path) {
            out.push(path);
        }
    }
}

// ---------------------------------------------------------------------------
// Application
// ---------------------------------------------------------------------------

struct App {
    tracks: Vec<Track>,
    selected: Option<usize>,
    player: Player,

    /// Path of the currently-open playlist file, if any. Persisted so it can
    /// be reloaded on the next launch.
    current_playlist: Option<PathBuf>,

    show_config: bool,
    bindings: Arc<Mutex<MidiBindings>>,
    action_tx: Sender<Action>,
    action_rx: Receiver<Action>,
    midi_conn: Option<MidiInputConnection<()>>,
    midi_ports: Vec<String>,
    connected_port: Option<String>,

    /// Index of the track currently loaded into the player (playing or paused),
    /// used to decide whether play/pause should resume or start the selection.
    playing_index: Option<usize>,

    /// Whether the one-time runtime "maximize" command has been sent.
    did_maximize: bool,
    status: String,
}

impl App {
    fn new(cc: &eframe::CreationContext<'_>) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        let (action_tx, action_rx) = channel();
        let mut app = Self {
            tracks: Vec::new(),
            selected: None,
            player: Player::new()?,
            current_playlist: None,
            show_config: false,
            bindings: Arc::new(Mutex::new(MidiBindings::default())),
            action_tx,
            action_rx,
            midi_conn: None,
            midi_ports: Vec::new(),
            connected_port: None,
            playing_index: None,
            did_maximize: false,
            status: String::new(),
        };

        if let Some(storage) = cc.storage {
            // Reload the last-loaded playlist, if it still exists.
            if let Some(path) = storage.get_string(LAST_PLAYLIST_KEY) {
                let path = PathBuf::from(path);
                if path.exists() {
                    app.load_playlist(&path);
                }
            }

            // Restore saved MIDI bindings.
            if let Some(text) = storage.get_string(MIDI_BINDINGS_KEY) {
                app.bindings.lock().unwrap().map = bindings_from_json(&text);
            }

            // Restore the last volume setting.
            if let Some(db) = storage.get_string(VOLUME_KEY).and_then(|s| s.parse().ok()) {
                app.player.set_volume_db(db);
            }

            // Reconnect to the last-used MIDI port if it is present.
            if let Some(port) = storage.get_string(MIDI_PORT_KEY) {
                app.refresh_ports();
                if app.midi_ports.contains(&port) {
                    app.connect_port(&cc.egui_ctx, &port);
                }
            }
        }

        Ok(app)
    }

    // --- track loading -----------------------------------------------------

    fn add_paths(&mut self, paths: Vec<PathBuf>) {
        let mut added = 0;
        for path in paths {
            if !is_audio_file(&path) {
                continue;
            }
            let name = path
                .file_name()
                .map(|n| n.to_string_lossy().into_owned())
                .unwrap_or_else(|| path.display().to_string());
            self.tracks.push(Track { path, name });
            added += 1;
        }
        if self.selected.is_none() && !self.tracks.is_empty() {
            self.selected = Some(0);
        }
        self.status = format!("Added {added} track(s)");
    }

    fn add_files_dialog(&mut self) {
        if let Some(files) = rfd::FileDialog::new()
            .add_filter("Audio", AUDIO_EXTENSIONS)
            .pick_files()
        {
            self.add_paths(files);
        }
    }

    fn add_folder_dialog(&mut self) {
        if let Some(dir) = rfd::FileDialog::new().pick_folder() {
            let mut found = Vec::new();
            scan_dir(&dir, &mut found);
            self.add_paths(found);
        }
    }

    // --- playlists ---------------------------------------------------------

    /// Replaces the track list with `paths`, rebuilding display names.
    fn set_tracks_from_paths(&mut self, paths: Vec<PathBuf>) {
        self.tracks = paths
            .into_iter()
            .map(|path| {
                let name = path
                    .file_name()
                    .map(|n| n.to_string_lossy().into_owned())
                    .unwrap_or_else(|| path.display().to_string());
                Track { path, name }
            })
            .collect();
        self.selected = if self.tracks.is_empty() { None } else { Some(0) };
    }

    /// Starts a new, empty playlist (does not touch playback).
    fn new_playlist(&mut self) {
        self.tracks.clear();
        self.selected = None;
        self.current_playlist = None;
        self.status = "New playlist".to_owned();
    }

    fn save_playlist_dialog(&mut self) {
        let default_name = self
            .current_playlist
            .as_ref()
            .and_then(|p| p.file_name())
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_else(|| "playlist.json".to_owned());
        let Some(path) = rfd::FileDialog::new()
            .add_filter("Playlist", &["json"])
            .set_file_name(default_name)
            .save_file()
        else {
            return;
        };

        let paths: Vec<String> = self
            .tracks
            .iter()
            .map(|t| t.path.to_string_lossy().into_owned())
            .collect();
        let json = match serde_json::to_string_pretty(&paths) {
            Ok(json) => json,
            Err(e) => {
                self.status = format!("Failed to serialize playlist: {e}");
                return;
            }
        };
        match std::fs::write(&path, json) {
            Ok(()) => {
                self.status = format!("Saved playlist: {}", path.display());
                self.current_playlist = Some(path);
            }
            Err(e) => self.status = format!("Save failed: {e}"),
        }
    }

    fn open_playlist_dialog(&mut self) {
        if let Some(path) = rfd::FileDialog::new()
            .add_filter("Playlist", &["json"])
            .pick_file()
        {
            self.load_playlist(&path);
        }
    }

    /// Loads a playlist (a JSON array of audio file paths) from `path`.
    fn load_playlist(&mut self, path: &Path) {
        let text = match std::fs::read_to_string(path) {
            Ok(text) => text,
            Err(e) => {
                self.status = format!("Open failed: {e}");
                return;
            }
        };
        match serde_json::from_str::<Vec<String>>(&text) {
            Ok(list) => {
                self.set_tracks_from_paths(list.into_iter().map(PathBuf::from).collect());
                self.status = format!("Loaded {} track(s) from {}", self.tracks.len(), path.display());
                self.current_playlist = Some(path.to_path_buf());
            }
            Err(e) => self.status = format!("Invalid playlist file: {e}"),
        }
    }

    // --- transport ---------------------------------------------------------

    fn dispatch(&mut self, action: Action) {
        match action {
            Action::PlayPause => self.play_pause(),
            Action::Stop => self.player.stop(),
            Action::Next => self.select_next(),
            Action::Prev => self.select_prev(),
        }
    }

    fn play_pause(&mut self) {
        match self.player.state {
            PlaybackState::Playing => self.player.pause(),
            // Resume only if the selection still points at the paused track;
            // otherwise start the newly-selected track.
            PlaybackState::Paused if self.selected == self.playing_index => self.player.resume(),
            PlaybackState::Paused | PlaybackState::Stopped => self.play_selected(),
        }
    }

    fn play_selected(&mut self) {
        let Some(idx) = self.selected else {
            self.status = "No track selected".to_owned();
            return;
        };
        let track = &self.tracks[idx];
        match self.player.play_path(&track.path, track.name.clone()) {
            Ok(()) => {
                self.playing_index = Some(idx);
                self.status = format!("Playing: {}", track.name);
            }
            Err(e) => self.status = format!("Failed to play {}: {e}", track.name),
        }
    }

    /// Next/previous only move the selection — never start playback.
    fn select_next(&mut self) {
        if self.tracks.is_empty() {
            return;
        }
        self.selected = Some(match self.selected {
            Some(i) => (i + 1).min(self.tracks.len() - 1),
            None => 0,
        });
    }

    fn select_prev(&mut self) {
        if self.tracks.is_empty() {
            return;
        }
        self.selected = Some(match self.selected {
            Some(i) => i.saturating_sub(1),
            None => 0,
        });
    }

    // --- playlist editing --------------------------------------------------
    //
    // These mutate the track list, so they must keep `selected` and
    // `playing_index` pointing at the same tracks — otherwise play/resume would
    // act on the wrong track.

    fn swap_tracks(&mut self, a: usize, b: usize) {
        self.tracks.swap(a, b);
        let remap = |idx: Option<usize>| match idx {
            Some(i) if i == a => Some(b),
            Some(i) if i == b => Some(a),
            other => other,
        };
        self.selected = remap(self.selected);
        self.playing_index = remap(self.playing_index);
    }

    fn move_up(&mut self, i: usize) {
        if i > 0 {
            self.swap_tracks(i, i - 1);
        }
    }

    fn move_down(&mut self, i: usize) {
        if i + 1 < self.tracks.len() {
            self.swap_tracks(i, i + 1);
        }
    }

    fn remove_track(&mut self, i: usize) {
        if i >= self.tracks.len() {
            return;
        }
        self.tracks.remove(i);
        let len = self.tracks.len();

        // Keep a sensible selection: same slot, clamped, or nothing if empty.
        self.selected = match self.selected {
            _ if len == 0 => None,
            Some(s) if s == i => Some(s.min(len - 1)),
            Some(s) if s > i => Some(s - 1),
            other => other,
        };

        // The loaded track keeps playing even if removed from the list, but it
        // is no longer addressable by index, so resume-by-selection is dropped.
        self.playing_index = match self.playing_index {
            Some(p) if p == i => None,
            Some(p) if p > i => Some(p - 1),
            other => other,
        };
    }

    // --- MIDI --------------------------------------------------------------

    fn refresh_ports(&mut self) {
        self.midi_ports = match MidiInput::new("clickinaresi-scan") {
            Ok(input) => input
                .ports()
                .iter()
                .map(|p| input.port_name(p).unwrap_or_else(|_| "<unknown>".to_owned()))
                .collect(),
            Err(_) => Vec::new(),
        };
    }

    fn connect_port(&mut self, ctx: &egui::Context, name: &str) {
        // Drop any existing connection first.
        self.midi_conn = None;
        self.connected_port = None;

        let input = match MidiInput::new("clickinaresi") {
            Ok(i) => i,
            Err(e) => {
                self.status = format!("MIDI error: {e}");
                return;
            }
        };
        let Some(port) = input
            .ports()
            .into_iter()
            .find(|p| input.port_name(p).as_deref() == Ok(name))
        else {
            self.status = format!("MIDI port not found: {name}");
            return;
        };

        let bindings = self.bindings.clone();
        let tx = self.action_tx.clone();
        let ctx = ctx.clone();
        match input.connect(
            &port,
            "clickinaresi-in",
            move |_stamp, msg, _| handle_midi(msg, &bindings, &tx, &ctx),
            (),
        ) {
            Ok(conn) => {
                self.midi_conn = Some(conn);
                self.connected_port = Some(name.to_owned());
                self.status = format!("Connected to MIDI: {name}");
            }
            Err(e) => self.status = format!("MIDI connect failed: {e}"),
        }
    }

    // --- UI ----------------------------------------------------------------

    fn transport_bar(&mut self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            let btn = |ui: &mut egui::Ui, glyph: &str| {
                ui.add_sized([72.0, 56.0], egui::Button::new(RichText::new(glyph).size(28.0)))
                    .clicked()
            };
            let toggle_glyph = if self.player.state == PlaybackState::Playing {
                "⏸"
            } else {
                "▶"
            };
            if btn(ui, toggle_glyph) {
                self.play_pause();
            }
            if btn(ui, "⏹") {
                self.player.stop();
            }
            if btn(ui, "⏮") {
                self.select_prev();
            }
            if btn(ui, "⏭") {
                self.select_next();
            }

            ui.add_space(16.0);
            ui.label(RichText::new("Vol").size(18.0));
            let mut db = self.player.volume_db;
            if ui
                .add(
                    egui::Slider::new(&mut db, -60.0..=24.0)
                        .suffix(" dB")
                        .fixed_decimals(0),
                )
                .changed()
            {
                self.player.set_volume_db(db);
            }

            ui.add_space(16.0);
            if ui.button(RichText::new("⚙ MIDI").size(18.0)).clicked() {
                self.show_config = true;
                self.refresh_ports();
            }
            if ui.button(RichText::new("Add files…").size(18.0)).clicked() {
                self.add_files_dialog();
            }
            if ui.button(RichText::new("Add folder…").size(18.0)).clicked() {
                self.add_folder_dialog();
            }

            ui.add_space(16.0);
            if ui.button(RichText::new("New").size(18.0)).clicked() {
                self.new_playlist();
            }
            if ui.button(RichText::new("Open…").size(18.0)).clicked() {
                self.open_playlist_dialog();
            }
            if ui.button(RichText::new("Save…").size(18.0)).clicked() {
                self.save_playlist_dialog();
            }
        });
    }

    fn now_playing(&self, ui: &mut egui::Ui) {
        let (caption, playing) = match (&self.player.current, self.player.state) {
            (Some(name), PlaybackState::Paused) => ("PAUSED", name.clone()),
            (Some(name), _) => ("NOW PLAYING", name.clone()),
            (None, _) => ("STOPPED", "—".to_owned()),
        };
        let selected = self
            .selected
            .map(|i| self.tracks[i].name.clone())
            .unwrap_or_else(|| "—".to_owned());

        const SIZE: f32 = 60.0;

        ui.horizontal(|ui| {
            ui.label(RichText::new(caption).size(16.0).weak());
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                ui.label(RichText::new("SELECTED").size(16.0).weak());
            });
        });
        ui.horizontal(|ui| {
            ui.label(RichText::new(first_chars(&playing, 15)).size(SIZE).strong());
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                ui.label(RichText::new(first_chars(&selected, 15)).size(SIZE).strong());
            });
        });
    }

    fn track_list(&mut self, ui: &mut egui::Ui) {
        if self.tracks.is_empty() {
            ui.add_space(20.0);
            ui.label(
                RichText::new("No tracks. Use “Add files…” or “Add folder…”.")
                    .size(20.0)
                    .weak(),
            );
            return;
        }
        // Structural edits are deferred until after the loop so we never mutate
        // the list while iterating it by index.
        enum Op {
            Up(usize),
            Down(usize),
            Remove(usize),
        }
        let n = self.tracks.len();
        let op = egui::ScrollArea::vertical()
            .auto_shrink([false, false])
            .show(ui, |ui| {
                let mut op: Option<Op> = None;
                for i in 0..n {
                    ui.horizontal(|ui| {
                        if ui
                            .add_enabled(i > 0, egui::Button::new(RichText::new("⏶").size(20.0)))
                            .clicked()
                        {
                            op = Some(Op::Up(i));
                        }
                        if ui
                            .add_enabled(
                                i + 1 < n,
                                egui::Button::new(RichText::new("⏷").size(20.0)),
                            )
                            .clicked()
                        {
                            op = Some(Op::Down(i));
                        }
                        if ui.button(RichText::new("×").size(20.0)).clicked() {
                            op = Some(Op::Remove(i));
                        }

                        let selected = self.selected == Some(i);
                        let text = RichText::new(&self.tracks[i].name).size(26.0);
                        let resp = ui.add(egui::SelectableLabel::new(selected, text));
                        if resp.clicked() {
                            self.selected = Some(i);
                        }
                        if resp.double_clicked() {
                            self.selected = Some(i);
                            self.player.stop();
                            self.play_selected();
                        }
                    });
                }
                op
            })
            .inner;

        match op {
            Some(Op::Up(i)) => self.move_up(i),
            Some(Op::Down(i)) => self.move_down(i),
            Some(Op::Remove(i)) => self.remove_track(i),
            None => {}
        }
    }

    fn config_window(&mut self, ctx: &egui::Context) {
        let mut open = self.show_config;
        egui::Window::new("MIDI Configuration")
            .open(&mut open)
            .resizable(true)
            .show(ctx, |ui| {
                ui.horizontal(|ui| {
                    ui.label("Input port:");
                    let current = self
                        .connected_port
                        .clone()
                        .unwrap_or_else(|| "(none)".to_owned());
                    egui::ComboBox::from_id_salt("midi_port")
                        .selected_text(current)
                        .show_ui(ui, |ui| {
                            let ports = self.midi_ports.clone();
                            for name in ports {
                                let selected = self.connected_port.as_deref() == Some(&name);
                                if ui.selectable_label(selected, &name).clicked() {
                                    self.connect_port(ctx, &name);
                                }
                            }
                        });
                    if ui.button("Refresh").clicked() {
                        self.refresh_ports();
                    }
                });

                ui.separator();
                ui.label(
                    RichText::new("Click “Learn”, then press a MIDI key or move a control.")
                        .weak(),
                );
                ui.add_space(4.0);

                let learning = self.bindings.lock().unwrap().learning;
                for action in Action::ALL {
                    ui.horizontal(|ui| {
                        ui.add_sized([110.0, 20.0], egui::Label::new(action.label()));

                        let binding = self.bindings.lock().unwrap().binding_for(action);
                        let binding_text = match (binding, learning == Some(action)) {
                            (_, true) => "← listening…".to_owned(),
                            (Some(key), _) => fmt_key(key),
                            (None, _) => "unassigned".to_owned(),
                        };
                        ui.add_sized([150.0, 20.0], egui::Label::new(binding_text));

                        if ui.button("Learn").clicked() {
                            let mut b = self.bindings.lock().unwrap();
                            b.learning = if b.learning == Some(action) {
                                None
                            } else {
                                Some(action)
                            };
                        }
                        if ui.button("Clear").clicked() {
                            self.bindings.lock().unwrap().clear(action);
                        }
                    });
                }
            });
        self.show_config = open;
    }
}

impl eframe::App for App {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // Force-maximize once on the first frame. `with_maximized` alone is not
        // reliably honored across platforms, so we also issue the command here.
        if !self.did_maximize {
            ctx.send_viewport_cmd(egui::ViewportCommand::Maximized(true));
            self.did_maximize = true;
        }

        // Drain MIDI-triggered actions queued by the input thread.
        while let Ok(action) = self.action_rx.try_recv() {
            self.dispatch(action);
        }
        self.player.poll();
        // Once playback is stopped (by the user or by reaching end-of-track),
        // there is no loaded track to resume.
        if self.player.state == PlaybackState::Stopped {
            self.playing_index = None;
        }

        egui::TopBottomPanel::top("now_playing").show(ctx, |ui| {
            ui.add_space(8.0);
            self.now_playing(ui);
            ui.add_space(8.0);
            self.transport_bar(ui);
            ui.add_space(8.0);
        });

        if !self.status.is_empty() {
            egui::TopBottomPanel::bottom("status").show(ctx, |ui| {
                ui.label(RichText::new(&self.status).weak());
            });
        }

        egui::CentralPanel::default().show(ctx, |ui| {
            self.track_list(ui);
        });

        if self.show_config {
            self.config_window(ctx);
        }

        // Keep polling the action queue and end-of-track detection responsive
        // without spinning the CPU when idle.
        ctx.request_repaint_after(Duration::from_millis(33));
    }

    /// Persists the path of the currently-open playlist so it can be reloaded
    /// on the next launch. eframe calls this periodically and on exit.
    fn save(&mut self, storage: &mut dyn Storage) {
        if let Some(path) = &self.current_playlist {
            storage.set_string(LAST_PLAYLIST_KEY, path.to_string_lossy().into_owned());
        }
        storage.set_string(
            MIDI_BINDINGS_KEY,
            bindings_to_json(&self.bindings.lock().unwrap().map),
        );
        if let Some(port) = &self.connected_port {
            storage.set_string(MIDI_PORT_KEY, port.clone());
        }
        storage.set_string(VOLUME_KEY, self.player.volume_db.to_string());
    }
}
