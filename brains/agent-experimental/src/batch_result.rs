pub struct BatchResult {
    initial: f32,
    after_first: f32,
    after_second: f32,
}

impl BatchResult {
    pub fn new(initial: f32, after_first: f32, after_second: f32) -> Self {
        Self {
            initial,
            after_first,
            after_second,
        }
    }

    pub fn initial(&self) -> f32 {
        self.initial
    }

    pub fn after_first(&self) -> f32 {
        self.after_first
    }

    pub fn after_second(&self) -> f32 {
        self.after_second
    }
}
