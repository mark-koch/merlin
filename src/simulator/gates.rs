use crate::Simulator;

impl Simulator {
    pub(crate) fn reset_z_raw(&mut self, qubit: usize) {
        if self.measure_z_raw(qubit).0 == -1 {
            self.x(qubit);
        }
    }

    pub(crate) fn reset_x_raw(&mut self, qubit: usize) {
        self.reset_z_raw(qubit);
        self.add_affine_coordinate(qubit, false);
    }

    /// Resets a qubit to |0>, growing the simulator as necessary.
    pub fn reset(&mut self, qubit: usize) {
        self.ensure_num_qubits(qubit + 1);
        self.reset_z_raw(qubit);
    }

    /// Alias of [`Self::reset`].
    pub fn reset_z(&mut self, qubit: usize) {
        self.reset(qubit);
    }

    /// Resets a qubit to |+>, growing the simulator as necessary.
    pub fn reset_x(&mut self, qubit: usize) {
        self.ensure_num_qubits(qubit + 1);
        self.reset_x_raw(qubit);
    }

    /// Applies X to a qubit, growing the simulator as necessary.
    pub fn x(&mut self, qubit: usize) {
        self.ensure_num_qubits(qubit + 1);
        self.x0.toggle(qubit);
    }

    /// Applies Y to a qubit, growing the simulator as necessary.
    pub fn y(&mut self, qubit: usize) {
        self.ensure_num_qubits(qubit + 1);
        self.x(qubit);
        self.z(qubit);
    }

    /// Applies Z to a qubit, growing the simulator as necessary.
    pub fn z(&mut self, qubit: usize) {
        self.ensure_num_qubits(qubit + 1);
        self.apply_t_power(qubit, 4);
    }

    /// Applies S to a qubit, growing the simulator as necessary.
    pub fn s(&mut self, qubit: usize) {
        self.ensure_num_qubits(qubit + 1);
        self.apply_t_power(qubit, 2);
    }

    /// Applies S-dagger to a qubit, growing the simulator as necessary.
    pub fn s_dag(&mut self, qubit: usize) {
        self.ensure_num_qubits(qubit + 1);
        self.apply_t_power(qubit, -2);
    }

    /// Applies T to a qubit, growing the simulator as necessary.
    pub fn t(&mut self, qubit: usize) {
        self.ensure_num_qubits(qubit + 1);
        self.apply_t_power(qubit, 1);
    }

    /// Applies T-dagger to a qubit, growing the simulator as necessary.
    pub fn t_dag(&mut self, qubit: usize) {
        self.ensure_num_qubits(qubit + 1);
        self.apply_t_power(qubit, -1);
    }

    /// Applies controlled-X, growing the simulator as necessary.
    ///
    /// # Panics
    /// Panics when the endpoints are equal.
    pub fn cnot(&mut self, control: usize, target: usize) {
        assert_ne!(control, target, "CNOT endpoints must be distinct");
        self.ensure_num_qubits(control.max(target) + 1);
        if self.x0.get(control) {
            self.x0.toggle(target);
        }
        let control_row = self.basis_rows[control].clone();
        self.basis_rows[target].xor_assign(&control_row);
    }

    /// Alias of [`Self::cnot`].
    pub fn cx(&mut self, control: usize, target: usize) {
        self.cnot(control, target);
    }

    /// Applies controlled-Z, growing the simulator as necessary.
    ///
    /// # Panics
    /// Panics when the endpoints are equal.
    pub fn cz(&mut self, left: usize, right: usize) {
        assert_ne!(left, right, "CZ endpoints must be distinct");
        self.ensure_num_qubits(left.max(right) + 1);
        self.apply_phase_gadget(&[left], 2);
        self.apply_phase_gadget(&[right], 2);
        self.apply_phase_gadget(&[left, right], -2);
    }

    /// Swaps two qubits, growing the simulator as necessary.
    ///
    /// # Panics
    /// Panics when the endpoints are equal.
    pub fn swap(&mut self, left: usize, right: usize) {
        assert_ne!(left, right, "SWAP endpoints must be distinct");
        self.ensure_num_qubits(left.max(right) + 1);
        let left_offset = self.x0.get(left);
        let right_offset = self.x0.get(right);
        self.x0.set(left, right_offset);
        self.x0.set(right, left_offset);
        self.basis_rows.swap(left, right);
    }

    /// Applies controlled-controlled-Z, growing the simulator as necessary.
    ///
    /// # Panics
    /// Panics unless the three endpoints are distinct.
    pub fn ccz(&mut self, first: usize, second: usize, third: usize) {
        assert!(
            first != second && first != third && second != third,
            "CCZ endpoints must be distinct"
        );
        self.ensure_num_qubits(first.max(second).max(third) + 1);
        self.apply_phase_gadget(&[first], 1);
        self.apply_phase_gadget(&[second], 1);
        self.apply_phase_gadget(&[third], 1);
        self.apply_phase_gadget(&[first, second], -1);
        self.apply_phase_gadget(&[first, third], -1);
        self.apply_phase_gadget(&[second, third], -1);
        self.apply_phase_gadget(&[first, second, third], 1);
    }

    pub(crate) fn apply_t_power(&mut self, qubit: usize, power: i16) {
        self.apply_phase_gadget(&[qubit], power);
    }
}

#[cfg(test)]
mod test {
    use crate::Simulator;

    #[test]
    fn gadgets_combine_and_cnot_does_not_rewrite_them() {
        let mut state = Simulator::from_product(b"++", 1);
        state.t(0);
        state.t_dag(0);
        assert!(state.gadgets.is_empty());
        state.t(0);
        let gadgets = state.gadgets.clone();
        state.cnot(0, 1);
        assert_eq!(state.gadgets, gadgets);
    }
}
