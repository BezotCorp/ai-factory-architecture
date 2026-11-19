#[derive(Clone, Copy, Debug)]
pub enum Strategy {
    Focused,
    Cooperative,
    Local,
}

impl Strategy {
    pub fn parse(value: &str) -> Option<Self> {
        match value {
            "focused" => Some(Self::Focused),
            "cooperative" => Some(Self::Cooperative),
            "local" => Some(Self::Local),
            _ => None,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::Focused => "focused",
            Self::Cooperative => "cooperative",
            Self::Local => "local",
        }
    }
}
