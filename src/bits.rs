#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub(crate) struct BitVec {
    n: usize,
    words: Vec<u64>,
}

impl BitVec {
    pub(crate) fn zero(n: usize) -> Self {
        Self {
            n,
            words: vec![0; n.div_ceil(64)],
        }
    }

    pub(crate) fn unit(n: usize, index: usize) -> Self {
        let mut out = Self::zero(n);
        out.set(index, true);
        out
    }

    pub(crate) fn len(&self) -> usize {
        self.n
    }

    pub(crate) fn resize(&mut self, new_len: usize) {
        self.n = new_len;
        self.words.resize(new_len.div_ceil(64), 0);
        if let Some(last) = self.words.last_mut()
            && !new_len.is_multiple_of(64)
        {
            *last &= (1_u64 << (new_len % 64)) - 1;
        }
    }

    pub(crate) fn get(&self, index: usize) -> bool {
        debug_assert!(index < self.n);
        (self.words[index / 64] >> (index % 64)) & 1 != 0
    }

    pub(crate) fn set(&mut self, index: usize, value: bool) {
        debug_assert!(index < self.n);
        let mask = 1_u64 << (index % 64);
        if value {
            self.words[index / 64] |= mask;
        } else {
            self.words[index / 64] &= !mask;
        }
    }

    pub(crate) fn toggle(&mut self, index: usize) {
        debug_assert!(index < self.n);
        self.words[index / 64] ^= 1_u64 << (index % 64);
    }

    pub(crate) fn xor_assign(&mut self, other: &Self) {
        debug_assert_eq!(self.n, other.n);
        for (left, right) in self.words.iter_mut().zip(&other.words) {
            *left ^= right;
        }
    }

    pub(crate) fn and_assign(&mut self, other: &Self) {
        debug_assert_eq!(self.n, other.n);
        for (left, right) in self.words.iter_mut().zip(&other.words) {
            *left &= right;
        }
    }

    pub(crate) fn xor(&self, other: &Self) -> Self {
        let mut out = self.clone();
        out.xor_assign(other);
        out
    }

    pub(crate) fn dot(&self, other: &Self) -> bool {
        debug_assert_eq!(self.n, other.n);
        self.words
            .iter()
            .zip(&other.words)
            .fold(0_u32, |parity, (a, b)| parity ^ (a & b).count_ones())
            & 1
            != 0
    }

    pub(crate) fn is_zero(&self) -> bool {
        self.words.iter().all(|word| *word == 0)
    }

    pub(crate) fn pivot(&self) -> Option<usize> {
        self.words
            .iter()
            .enumerate()
            .find_map(|(word_index, word)| {
                (*word != 0).then(|| word_index * 64 + word.trailing_zeros() as usize)
            })
    }

    pub(crate) fn iter_ones(&self) -> impl Iterator<Item = usize> + '_ {
        self.words
            .iter()
            .copied()
            .enumerate()
            .flat_map(|(word_index, mut word)| {
                std::iter::from_fn(move || {
                    if word == 0 {
                        return None;
                    }
                    let bit = word.trailing_zeros() as usize;
                    word &= word - 1;
                    Some(word_index * 64 + bit)
                })
            })
            .take_while(|index| *index < self.n)
    }

    pub(crate) fn to_index_msb(&self) -> usize {
        let mut index = 0;
        for qubit in 0..self.n {
            index |= usize::from(self.get(qubit)) << (self.n - 1 - qubit);
        }
        index
    }

    pub(crate) fn to_index_lsb(&self) -> usize {
        let mut index = 0;
        for qubit in 0..self.n {
            index |= usize::from(self.get(qubit)) << qubit;
        }
        index
    }

    pub(crate) fn remove_bit(&self, idx: usize) -> BitVec {
        let mut out = BitVec::zero(self.len() - 1);
        for old in 0..self.len() {
            if old != idx && self.get(old) {
                out.set(if old < idx { old } else { old - 1 }, true);
            }
        }
        out
    }
}
