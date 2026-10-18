//! Undo/redo and A/B slots, implemented as snapshots of every parameter.

use nih_plug::prelude::*;

const MAX_UNDO: usize = 200;
/// Edits closer together than this (and not during a mouse drag) merge into one step.
const SETTLE_S: f64 = 0.3;

pub struct History {
    params: Vec<ParamPtr>,
    last: Vec<f32>,
    pending: Option<Vec<f32>>,
    last_change: f64,
    ignore_frames: u8,
    undo: Vec<Vec<f32>>,
    redo: Vec<Vec<f32>>,
    pub ab_active: usize,
    slots: [Option<Vec<f32>>; 2],
}

impl History {
    pub fn new(params: &dyn Params) -> Self {
        let ptrs: Vec<ParamPtr> = params.param_map().into_iter().map(|(_, p, _)| p).collect();
        let mut h = Self {
            params: ptrs,
            last: Vec::new(),
            pending: None,
            last_change: 0.0,
            ignore_frames: 0,
            undo: Vec::new(),
            redo: Vec::new(),
            ab_active: 0,
            slots: [None, None],
        };
        h.last = h.snapshot();
        h
    }

    fn snapshot(&self) -> Vec<f32> {
        // Safety: the pointers come from the live params object, which outlives the editor.
        self.params
            .iter()
            .map(|p| unsafe { p.unmodulated_normalized_value() })
            .collect()
    }

    fn apply(&mut self, setter: &ParamSetter, target: &[f32]) {
        for (p, &v) in self.params.iter().zip(target) {
            let cur = unsafe { p.unmodulated_normalized_value() };
            if (cur - v).abs() > f32::EPSILON {
                unsafe {
                    setter.raw_context.raw_begin_set_parameter(*p);
                    setter.raw_context.raw_set_parameter_normalized(*p, v);
                    setter.raw_context.raw_end_set_parameter(*p);
                }
            }
        }
        self.last = target.to_vec();
        self.ignore_frames = 2;
        self.pending = None;
    }

    /// Call once per GUI frame to record edits.
    pub fn update(&mut self, now: f64, pointer_down: bool) {
        if self.ignore_frames > 0 {
            self.ignore_frames -= 1;
            return;
        }
        let cur = self.snapshot();
        if cur != self.last {
            if self.pending.is_none() {
                self.pending = Some(std::mem::replace(&mut self.last, Vec::new()));
            }
            self.last = cur;
            self.last_change = now;
        }
        if self.pending.is_some() && !pointer_down && now - self.last_change > SETTLE_S {
            self.commit();
        }
    }

    fn commit(&mut self) {
        if let Some(p) = self.pending.take() {
            self.undo.push(p);
            if self.undo.len() > MAX_UNDO {
                self.undo.remove(0);
            }
            self.redo.clear();
        }
    }

    pub fn can_undo(&self) -> bool {
        !self.undo.is_empty() || self.pending.is_some()
    }

    pub fn can_redo(&self) -> bool {
        !self.redo.is_empty()
    }

    pub fn undo(&mut self, setter: &ParamSetter) {
        self.commit();
        if let Some(target) = self.undo.pop() {
            let cur = self.snapshot();
            self.redo.push(cur);
            self.apply(setter, &target);
        }
    }

    pub fn redo(&mut self, setter: &ParamSetter) {
        if let Some(target) = self.redo.pop() {
            let cur = self.snapshot();
            self.undo.push(cur);
            self.apply(setter, &target);
        }
    }

    /// Switch to A (0) or B (1): stash the current state in the active slot and load the other.
    pub fn switch_ab(&mut self, setter: &ParamSetter, to: usize) {
        if to == self.ab_active {
            return;
        }
        self.commit();
        let cur = self.snapshot();
        let target = self.slots[to].clone().unwrap_or_else(|| cur.clone());
        self.slots[self.ab_active] = Some(cur);
        self.ab_active = to;
        self.apply(setter, &target);
    }
}
