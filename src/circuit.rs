use pest::Parser;
use pest::iterators::Pair;
use pest_derive::Parser;
use std::collections::HashSet;

#[derive(Parser)]
#[grammar = "stim.pest"]
struct StimParser;

#[derive(Clone, Debug)]
struct AstInstruction {
    name: String,
    arguments: Vec<f64>,
    targets: Vec<String>,
}

#[derive(Clone, Debug)]
enum AstNode {
    Instruction(AstInstruction),
    Repeat { count: usize, body: Vec<AstNode> },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Basis {
    X,
    Z,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Pauli {
    X,
    Y,
    Z,
}

impl Pauli {
    pub(crate) fn code(self) -> u8 {
        match self {
            Self::X => 1,
            Self::Y => 2,
            Self::Z => 3,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct PauliProduct {
    pub(crate) factors: Vec<(usize, Pauli)>,
    pub(crate) negative: bool,
    pub(crate) num_qubits: usize,
}

impl PauliProduct {
    pub(crate) fn from_compact(source: &str) -> Result<Self, String> {
        let (negative, body) = if let Some(body) = source.strip_prefix('-') {
            (true, body)
        } else if let Some(body) = source.strip_prefix('+') {
            (false, body)
        } else {
            (false, source)
        };
        if body.is_empty() {
            return Err("compact Pauli observable must contain at least one factor".to_owned());
        }
        let mut factors = Vec::new();
        let mut num_qubits = 0;
        for (qubit, symbol) in body.chars().enumerate() {
            num_qubits += 1;
            let pauli = match symbol {
                '_' | 'I' => continue,
                'X' => Pauli::X,
                'Y' => Pauli::Y,
                'Z' => Pauli::Z,
                _ => {
                    return Err(format!(
                        "invalid compact Pauli observable symbol {symbol:?} at position {qubit}"
                    ));
                }
            };
            factors.push((qubit, pauli));
        }
        Ok(Self {
            factors,
            negative,
            num_qubits,
        })
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum GateInstruction {
    I(Vec<usize>),
    X(Vec<usize>),
    Y(Vec<usize>),
    Z(Vec<usize>),
    S(Vec<usize>),
    Sdg(Vec<usize>),
    T(Vec<usize>),
    Tdg(Vec<usize>),
    Cx(Vec<(usize, usize)>),
    Cz(Vec<(usize, usize)>),
    Ccz(Vec<(usize, usize, usize)>),
    Swap(Vec<(usize, usize)>),
}

impl GateInstruction {
    pub(crate) fn for_each_target(&self, mut visit: impl FnMut(usize)) {
        match self {
            Self::I(qubits)
            | Self::X(qubits)
            | Self::Y(qubits)
            | Self::Z(qubits)
            | Self::S(qubits)
            | Self::Sdg(qubits)
            | Self::T(qubits)
            | Self::Tdg(qubits) => qubits.iter().copied().for_each(&mut visit),
            Self::Cx(pairs) | Self::Cz(pairs) | Self::Swap(pairs) => {
                for &(left, right) in pairs {
                    visit(left);
                    visit(right);
                }
            }
            Self::Ccz(triples) => {
                for &(first, second, third) in triples {
                    visit(first);
                    visit(second);
                    visit(third);
                }
            }
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) enum NoiseInstruction {
    Error {
        pauli: Pauli,
        qubits: Vec<usize>,
        probability: f64,
    },
    Depolarize1 {
        qubits: Vec<usize>,
        probability: f64,
    },
    Depolarize2 {
        pairs: Vec<(usize, usize)>,
        probability: f64,
    },
    Depolarize3 {
        triples: Vec<(usize, usize, usize)>,
        probability: f64,
    },
    PauliChannel1 {
        qubits: Vec<usize>,
        probabilities: [f64; 3],
    },
    PauliChannel2 {
        pairs: Vec<(usize, usize)>,
        probabilities: [f64; 15],
    },
}

impl NoiseInstruction {
    pub(crate) fn num_fault_sites(&self) -> usize {
        match self {
            Self::Error { qubits, .. }
            | Self::Depolarize1 { qubits, .. }
            | Self::PauliChannel1 { qubits, .. } => qubits.len(),
            Self::Depolarize2 { pairs, .. } | Self::PauliChannel2 { pairs, .. } => pairs.len(),
            Self::Depolarize3 { triples, .. } => triples.len(),
        }
    }

    pub(crate) fn for_each_target(&self, mut visit: impl FnMut(usize)) {
        match self {
            Self::Error { qubits, .. }
            | Self::Depolarize1 { qubits, .. }
            | Self::PauliChannel1 { qubits, .. } => {
                qubits.iter().copied().for_each(&mut visit);
            }
            Self::Depolarize2 { pairs, .. } | Self::PauliChannel2 { pairs, .. } => {
                for &(left, right) in pairs {
                    visit(left);
                    visit(right);
                }
            }
            Self::Depolarize3 { triples, .. } => {
                for &(first, second, third) in triples {
                    visit(first);
                    visit(second);
                    visit(third);
                }
            }
        }
    }
}

#[derive(Clone, Debug)]
pub(crate) enum Instruction {
    Gate(GateInstruction),
    ClassicallyControlledPauli {
        pauli: Pauli,
        lookback: usize,
        target: usize,
    },
    Measure {
        basis: Basis,
        qubits: Vec<(usize, bool)>,
        readout_probability: Option<f64>,
        reset: bool,
    },
    MeasurePauliProduct {
        products: Vec<PauliProduct>,
        readout_probability: Option<f64>,
    },
    ExpectationValue {
        products: Vec<PauliProduct>,
    },
    Reset {
        basis: Basis,
        qubits: Vec<usize>,
    },
    Noise(NoiseInstruction),
    Detector {
        lookbacks: Vec<usize>,
    },
    ObservableInclude {
        observable: usize,
        lookbacks: Vec<usize>,
    },
}

impl Instruction {
    fn for_each_target(&self, mut visit: impl FnMut(usize)) {
        match self {
            Self::Gate(gate) => gate.for_each_target(visit),
            Self::ClassicallyControlledPauli { target, .. } => visit(*target),
            Self::Measure { qubits, .. } => {
                qubits.iter().for_each(|(qubit, _)| visit(*qubit));
            }
            Self::MeasurePauliProduct { products, .. } => {
                for product in products {
                    product.factors.iter().for_each(|(qubit, _)| visit(*qubit));
                }
            }
            Self::ExpectationValue { products } => {
                for product in products {
                    product.factors.iter().for_each(|(qubit, _)| visit(*qubit));
                }
            }
            Self::Reset { qubits, .. } => qubits.iter().copied().for_each(visit),
            Self::Noise(noise) => noise.for_each_target(visit),
            Self::Detector { .. } | Self::ObservableInclude { .. } => {}
        }
    }
}

/// A parsed and validated Stim-format circuit.
///
/// The instruction representation is intentionally opaque.
#[derive(Clone, Debug)]
pub struct Circuit {
    pub(crate) num_qubits: usize,
    pub(crate) instructions: Vec<Instruction>,
}

impl Circuit {
    /// Parses a Stim-format source string.
    ///
    /// ```
    /// use merlin::Circuit;
    /// let circuit = Circuit::parse("RX 0\nT 0\nMX 0");
    /// assert_eq!(circuit.num_qubits(), 1);
    /// assert_eq!(circuit.num_measurements(), 1);
    /// ```
    ///
    /// # Panics
    /// Panics when the source is malformed or contains an unsupported operation.
    pub fn parse(source: &str) -> Self {
        Self::try_parse(source).unwrap_or_else(|message| panic!("{message}"))
    }

    pub(crate) fn try_parse(source: &str) -> Result<Self, String> {
        let file = StimParser::parse(Rule::file, source)
            .map_err(|error| format!("invalid Stim circuit: {error}"))?
            .next()
            .ok_or_else(|| "invalid Stim circuit: parser returned no file".to_owned())?;
        let nodes = parse_nodes(file)?;
        validate_nodes(&nodes)?;

        let mut ast_instructions = Vec::new();
        expand(&nodes, &mut ast_instructions);

        let mut instructions = Vec::new();
        for instruction in &ast_instructions {
            push_instruction(instruction, &mut instructions)?;
        }
        validate_record_lookbacks(&instructions)?;
        Ok(Self::from_instructions(instructions))
    }

    /// Returns the number of qubits referenced by the circuit.
    pub fn num_qubits(&self) -> usize {
        self.num_qubits
    }

    /// Returns the number of measurement results produced by the circuit.
    pub fn num_measurements(&self) -> usize {
        self.instructions
            .iter()
            .map(|instruction| match instruction {
                Instruction::Measure { qubits, .. } => qubits.len(),
                Instruction::MeasurePauliProduct { products, .. } => products.len(),
                _ => 0,
            })
            .sum()
    }

    /// Returns the number of detectors declared by the circuit.
    pub fn num_detectors(&self) -> usize {
        self.instructions
            .iter()
            .filter(|instruction| matches!(instruction, Instruction::Detector { .. }))
            .count()
    }

    /// Returns the number of logical observables declared by the circuit.
    pub fn num_observables(&self) -> usize {
        self.instructions
            .iter()
            .filter_map(|instruction| match instruction {
                Instruction::ObservableInclude { observable, .. } => Some(observable + 1),
                _ => None,
            })
            .max()
            .unwrap_or(0)
    }

    /// Returns the number of expectation-value columns produced by the circuit.
    pub fn num_exp_vals(&self) -> usize {
        self.instructions
            .iter()
            .map(|instruction| match instruction {
                Instruction::ExpectationValue { products } => products.len(),
                _ => 0,
            })
            .sum()
    }

    pub(crate) fn num_fault_sites(&self) -> usize {
        self.instructions
            .iter()
            .map(|instruction| match instruction {
                Instruction::Noise(noise) => noise.num_fault_sites(),
                Instruction::Measure {
                    qubits,
                    readout_probability: Some(_),
                    ..
                } => qubits.len(),
                Instruction::MeasurePauliProduct {
                    products,
                    readout_probability: Some(_),
                } => products.len(),
                _ => 0,
            })
            .sum()
    }

    pub(crate) fn contains_pauli_channel(&self) -> bool {
        self.instructions.iter().any(|instruction| {
            matches!(
                instruction,
                Instruction::Noise(
                    NoiseInstruction::PauliChannel1 { .. } | NoiseInstruction::PauliChannel2 { .. }
                )
            )
        })
    }

    pub(crate) fn from_instructions(instructions: Vec<Instruction>) -> Self {
        let mut num_qubits = 0;
        for instruction in &instructions {
            instruction.for_each_target(|target| {
                num_qubits = num_qubits.max(target + 1);
            });
        }
        Self {
            num_qubits,
            instructions,
        }
    }
}

fn validate_record_lookbacks(instructions: &[Instruction]) -> Result<(), String> {
    let mut measurements = 0;
    for instruction in instructions {
        match instruction {
            Instruction::Measure { qubits, .. } => measurements += qubits.len(),
            Instruction::MeasurePauliProduct { products, .. } => measurements += products.len(),
            Instruction::ExpectationValue { .. } => {}
            Instruction::ClassicallyControlledPauli { lookback, .. }
                if *lookback > measurements =>
            {
                return Err(format!(
                    "record target rec[-{lookback}] refers before the beginning of the measurement record"
                ));
            }
            Instruction::Detector { lookbacks }
            | Instruction::ObservableInclude { lookbacks, .. } => {
                if let Some(&lookback) = lookbacks.iter().find(|&&value| value > measurements) {
                    return Err(format!(
                        "record target rec[-{lookback}] refers before the beginning of the measurement record"
                    ));
                }
            }
            _ => {}
        }
    }
    Ok(())
}

fn parse_nodes(pair: Pair<'_, Rule>) -> Result<Vec<AstNode>, String> {
    let mut nodes = Vec::new();
    for child in pair.into_inner() {
        match child.as_rule() {
            Rule::instruction => nodes.push(AstNode::Instruction(parse_instruction(child)?)),
            Rule::repeat_block => nodes.push(parse_repeat(child)?),
            _ => {}
        }
    }
    Ok(nodes)
}

fn parse_repeat(pair: Pair<'_, Rule>) -> Result<AstNode, String> {
    let mut inner = pair.into_inner();
    let count = inner
        .next()
        .ok_or_else(|| "REPEAT is missing its count".to_owned())?
        .as_str()
        .parse::<usize>()
        .map_err(|_| "REPEAT count does not fit in usize".to_owned())?;
    let mut body = Vec::new();
    for item in inner {
        match item.as_rule() {
            Rule::instruction => body.push(AstNode::Instruction(parse_instruction(item)?)),
            Rule::repeat_block => body.push(parse_repeat(item)?),
            _ => {}
        }
    }
    Ok(AstNode::Repeat { count, body })
}

fn parse_instruction(pair: Pair<'_, Rule>) -> Result<AstInstruction, String> {
    let mut inner = pair.into_inner();
    let name = inner
        .next()
        .ok_or_else(|| "instruction is missing its name".to_owned())?
        .as_str()
        .to_ascii_uppercase();
    let mut arguments = Vec::new();
    let mut targets = Vec::new();
    for item in inner {
        match item.as_rule() {
            Rule::arguments => {
                for number in item.into_inner() {
                    arguments.push(
                        number.as_str().parse::<f64>().map_err(|_| {
                            format!("invalid numeric argument {:?}", number.as_str())
                        })?,
                    );
                }
            }
            Rule::target => targets.push(item.as_str().to_owned()),
            _ => {}
        }
    }
    Ok(AstInstruction {
        name,
        arguments,
        targets,
    })
}

fn validate_nodes(nodes: &[AstNode]) -> Result<(), String> {
    let mut scratch = Vec::new();
    for node in nodes {
        match node {
            AstNode::Instruction(instruction) => {
                push_instruction(instruction, &mut scratch)?;
                scratch.clear();
            }
            AstNode::Repeat { body, .. } => validate_nodes(body)?,
        }
    }
    Ok(())
}

fn expand(nodes: &[AstNode], out: &mut Vec<AstInstruction>) {
    for node in nodes {
        match node {
            AstNode::Instruction(instruction) => out.push(instruction.clone()),
            AstNode::Repeat { count, body } => {
                for _ in 0..*count {
                    expand(body, out);
                }
            }
        }
    }
}

fn probability(value: f64) -> Result<f64, String> {
    if value.is_finite() && (0.0..=1.0).contains(&value) {
        Ok(value)
    } else {
        Err(format!(
            "probability must be finite and between 0 and 1, but got {value}"
        ))
    }
}

fn probabilities(arguments: &[f64], expected: usize, gate: &str) -> Result<Vec<f64>, String> {
    if arguments.len() != expected {
        return Err(format!(
            "{gate} requires {expected} probability argument(s), but got {}",
            arguments.len()
        ));
    }
    let values = arguments
        .iter()
        .map(|value| probability(*value))
        .collect::<Result<Vec<_>, _>>()?;
    if values.iter().sum::<f64>() > 1.0 {
        return Err(format!("{gate} probabilities sum to more than 1"));
    }
    Ok(values)
}

fn no_arguments(instruction: &AstInstruction) -> Result<(), String> {
    if instruction.arguments.is_empty() {
        Ok(())
    } else {
        Err(format!("{} does not take arguments", instruction.name))
    }
}

fn measurement_probability(instruction: &AstInstruction) -> Result<Option<f64>, String> {
    match instruction.arguments.as_slice() {
        [] => Ok(None),
        [value] => probability(*value).map(Some),
        _ => Err(format!(
            "{} takes at most one readout probability",
            instruction.name
        )),
    }
}

fn qubit_target(raw: &str, allow_inversion: bool) -> Result<(usize, bool), String> {
    let (inverted, text) = raw
        .strip_prefix('!')
        .map_or((false, raw), |text| (true, text));
    if inverted && !allow_inversion {
        return Err(format!(
            "inverted target {raw:?} is only valid for measurements"
        ));
    }
    if text.starts_with("rec[") || text.starts_with("sweep[") {
        return Err(format!(
            "measurement feedback target {raw:?} is unsupported"
        ));
    }
    let qubit = text
        .parse::<usize>()
        .map_err(|_| format!("invalid qubit target {raw:?}"))?;
    Ok((qubit, inverted))
}

fn qubits(instruction: &AstInstruction) -> Result<Vec<usize>, String> {
    instruction
        .targets
        .iter()
        .map(|target| qubit_target(target, false).map(|target| target.0))
        .collect()
}

fn measured_qubits(instruction: &AstInstruction) -> Result<Vec<(usize, bool)>, String> {
    instruction
        .targets
        .iter()
        .map(|target| qubit_target(target, true))
        .collect()
}

fn record_lookbacks(instruction: &AstInstruction) -> Result<Vec<usize>, String> {
    instruction
        .targets
        .iter()
        .map(|target| record_lookback(target))
        .collect()
}

fn record_lookback(target: &str) -> Result<usize, String> {
    let value = target
        .strip_prefix("rec[-")
        .and_then(|value| value.strip_suffix(']'))
        .ok_or_else(|| format!("invalid measurement record target {target:?}"))?;
    let lookback = value
        .parse::<usize>()
        .map_err(|_| format!("invalid measurement record target {target:?}"))?;
    if lookback == 0 {
        return Err("measurement record lookback must be positive".to_owned());
    }
    Ok(lookback)
}

fn observable_index(instruction: &AstInstruction) -> Result<usize, String> {
    let [value] = instruction.arguments.as_slice() else {
        return Err("OBSERVABLE_INCLUDE requires one observable index".to_owned());
    };
    if !value.is_finite() || *value < 0.0 || value.fract() != 0.0 {
        return Err(format!(
            "OBSERVABLE_INCLUDE index must be a non-negative integer, but got {value}"
        ));
    }
    let index = *value as usize;
    if index as f64 != *value {
        return Err(format!("OBSERVABLE_INCLUDE index {value} is too large"));
    }
    Ok(index)
}

fn pauli_product(raw: &str, instruction: &str) -> Result<PauliProduct, String> {
    if raw.is_empty() {
        return Err(format!("{instruction} product cannot be empty"));
    }
    let mut factors = Vec::new();
    let mut negative = false;
    let mut seen = HashSet::new();
    let mut num_qubits = 0;
    for raw_factor in raw.split('*') {
        if raw_factor.is_empty() {
            return Err(format!("malformed {instruction} product {raw:?}"));
        }
        let (inverted, factor) = raw_factor
            .strip_prefix('!')
            .map_or((false, raw_factor), |factor| (true, factor));
        negative ^= inverted;
        let mut chars = factor.chars();
        let pauli = match chars.next() {
            Some('X') => Pauli::X,
            Some('Y') => Pauli::Y,
            Some('Z') => Pauli::Z,
            _ => return Err(format!("invalid {instruction} Pauli factor {raw_factor:?}")),
        };
        let index = chars.as_str();
        if index.is_empty() || !index.bytes().all(|byte| byte.is_ascii_digit()) {
            return Err(format!("invalid {instruction} Pauli factor {raw_factor:?}"));
        }
        let qubit = index
            .parse::<usize>()
            .map_err(|_| format!("invalid {instruction} qubit index in {raw_factor:?}"))?;
        if !seen.insert(qubit) {
            return Err(format!(
                "{instruction} product {raw:?} contains qubit {qubit} more than once"
            ));
        }
        num_qubits = num_qubits.max(qubit + 1);
        factors.push((qubit, pauli));
    }
    Ok(PauliProduct {
        factors,
        negative,
        num_qubits,
    })
}

fn paired_qubits(instruction: &AstInstruction) -> Result<Vec<(usize, usize)>, String> {
    let targets = qubits(instruction)?;
    if !targets.len().is_multiple_of(2) {
        return Err(format!(
            "{} requires an even number of targets",
            instruction.name
        ));
    }
    let mut pairs = Vec::with_capacity(targets.len() / 2);
    for pair in targets.chunks_exact(2) {
        if pair[0] == pair[1] {
            return Err(format!(
                "{} pair endpoints must be different (both were {})",
                instruction.name, pair[0]
            ));
        }
        pairs.push((pair[0], pair[1]));
    }
    Ok(pairs)
}

fn tripled_qubits(instruction: &AstInstruction) -> Result<Vec<(usize, usize, usize)>, String> {
    let targets = qubits(instruction)?;
    if !targets.len().is_multiple_of(3) {
        return Err(format!(
            "{} requires a multiple of three targets",
            instruction.name
        ));
    }
    let mut triples = Vec::with_capacity(targets.len() / 3);
    for triple in targets.chunks_exact(3) {
        if triple[0] == triple[1] || triple[0] == triple[2] || triple[1] == triple[2] {
            return Err(format!(
                "{} triple endpoints must be different (got {}, {}, {})",
                instruction.name, triple[0], triple[1], triple[2]
            ));
        }
        triples.push((triple[0], triple[1], triple[2]));
    }
    Ok(triples)
}

fn push_controlled_pairs(
    instruction: &AstInstruction,
    out: &mut Vec<Instruction>,
) -> Result<(), String> {
    if !instruction.targets.len().is_multiple_of(2) {
        return Err(format!(
            "{} requires an even number of targets",
            instruction.name
        ));
    }
    for pair in instruction.targets.chunks_exact(2) {
        if pair[0].starts_with("rec[") {
            let lookback = record_lookback(&pair[0])?;
            let target = qubit_target(&pair[1], false)?.0;
            let pauli = match instruction.name.as_str() {
                "CX" | "CNOT" => Pauli::X,
                "CY" => Pauli::Y,
                "CZ" => Pauli::Z,
                _ => unreachable!(),
            };
            out.push(Instruction::ClassicallyControlledPauli {
                pauli,
                lookback,
                target,
            });
            continue;
        }

        let control = qubit_target(&pair[0], false)?.0;
        let target = qubit_target(&pair[1], false)?.0;
        if instruction.name == "CY" {
            return Err("ordinary qubit-controlled CY is unsupported".to_owned());
        }
        if control == target {
            return Err(format!(
                "{} pair endpoints must be different (both were {control})",
                instruction.name
            ));
        }
        out.push(Instruction::Gate(match instruction.name.as_str() {
            "CX" | "CNOT" => GateInstruction::Cx(vec![(control, target)]),
            "CZ" => GateInstruction::Cz(vec![(control, target)]),
            _ => unreachable!(),
        }));
    }
    Ok(())
}

fn push_instruction(
    instruction: &AstInstruction,
    out: &mut Vec<Instruction>,
) -> Result<(), String> {
    let name = instruction.name.as_str();
    if matches!(name, "TICK" | "QUBIT_COORDS" | "SHIFT_COORDS") {
        return Ok(());
    }

    match name {
        "I" | "X" | "Y" | "Z" | "S" | "S_DAG" | "T" | "T_DAG" => {
            no_arguments(instruction)?;
            let qubits = qubits(instruction)?;
            out.push(Instruction::Gate(match name {
                "I" => GateInstruction::I(qubits),
                "X" => GateInstruction::X(qubits),
                "Y" => GateInstruction::Y(qubits),
                "Z" => GateInstruction::Z(qubits),
                "S" => GateInstruction::S(qubits),
                "S_DAG" => GateInstruction::Sdg(qubits),
                "T" => GateInstruction::T(qubits),
                "T_DAG" => GateInstruction::Tdg(qubits),
                _ => unreachable!(),
            }));
        }
        "CX" | "CNOT" | "CY" | "CZ" => {
            no_arguments(instruction)?;
            push_controlled_pairs(instruction, out)?;
        }
        "SWAP" => {
            no_arguments(instruction)?;
            out.push(Instruction::Gate(GateInstruction::Swap(paired_qubits(
                instruction,
            )?)));
        }
        "CCZ" => {
            no_arguments(instruction)?;
            out.push(Instruction::Gate(GateInstruction::Ccz(tripled_qubits(
                instruction,
            )?)));
        }
        "M" | "MZ" | "MX" | "MR" | "MRZ" | "MRX" => {
            let basis = if matches!(name, "MX" | "MRX") {
                Basis::X
            } else {
                Basis::Z
            };
            out.push(Instruction::Measure {
                basis,
                qubits: measured_qubits(instruction)?,
                readout_probability: measurement_probability(instruction)?,
                reset: matches!(name, "MR" | "MRZ" | "MRX"),
            });
        }
        "MPP" => {
            let readout_probability = measurement_probability(instruction)?;
            let products = instruction
                .targets
                .iter()
                .map(|target| pauli_product(target, "MPP"))
                .collect::<Result<Vec<_>, _>>()?;
            out.push(Instruction::MeasurePauliProduct {
                products,
                readout_probability,
            });
        }
        "EXP_VAL" => {
            no_arguments(instruction)?;
            if instruction.targets.is_empty() {
                return Err("EXP_VAL requires at least one Pauli product".to_owned());
            }
            let products = instruction
                .targets
                .iter()
                .map(|target| pauli_product(target, "EXP_VAL"))
                .collect::<Result<Vec<_>, _>>()?;
            out.push(Instruction::ExpectationValue { products });
        }
        "DETECTOR" => {
            out.push(Instruction::Detector {
                lookbacks: record_lookbacks(instruction)?,
            });
        }
        "OBSERVABLE_INCLUDE" => {
            out.push(Instruction::ObservableInclude {
                observable: observable_index(instruction)?,
                lookbacks: record_lookbacks(instruction)?,
            });
        }
        "R" | "RZ" | "RX" => {
            no_arguments(instruction)?;
            out.push(Instruction::Reset {
                basis: if name == "RX" { Basis::X } else { Basis::Z },
                qubits: qubits(instruction)?,
            });
        }
        "X_ERROR" | "Y_ERROR" | "Z_ERROR" => {
            let probability = probabilities(&instruction.arguments, 1, name)?[0];
            out.push(Instruction::Noise(NoiseInstruction::Error {
                pauli: match name {
                    "X_ERROR" => Pauli::X,
                    "Y_ERROR" => Pauli::Y,
                    "Z_ERROR" => Pauli::Z,
                    _ => unreachable!(),
                },
                qubits: qubits(instruction)?,
                probability,
            }));
        }
        "DEPOLARIZE1" => {
            out.push(Instruction::Noise(NoiseInstruction::Depolarize1 {
                qubits: qubits(instruction)?,
                probability: probabilities(&instruction.arguments, 1, name)?[0],
            }));
        }
        "DEPOLARIZE2" => {
            out.push(Instruction::Noise(NoiseInstruction::Depolarize2 {
                pairs: paired_qubits(instruction)?,
                probability: probabilities(&instruction.arguments, 1, name)?[0],
            }));
        }
        "DEPOLARIZE3" => {
            out.push(Instruction::Noise(NoiseInstruction::Depolarize3 {
                triples: tripled_qubits(instruction)?,
                probability: probabilities(&instruction.arguments, 1, name)?[0],
            }));
        }
        "PAULI_CHANNEL_1" => {
            let probabilities: [f64; 3] = probabilities(&instruction.arguments, 3, name)?
                .try_into()
                .map_err(|_| "invalid PAULI_CHANNEL_1 probability count".to_owned())?;
            out.push(Instruction::Noise(NoiseInstruction::PauliChannel1 {
                qubits: qubits(instruction)?,
                probabilities,
            }));
        }
        "PAULI_CHANNEL_2" => {
            let probabilities: [f64; 15] = probabilities(&instruction.arguments, 15, name)?
                .try_into()
                .map_err(|_| "invalid PAULI_CHANNEL_2 probability count".to_owned())?;
            out.push(Instruction::Noise(NoiseInstruction::PauliChannel2 {
                pairs: paired_qubits(instruction)?,
                probabilities,
            }));
        }
        _ => return Err(format!("unsupported instruction {}", instruction.name)),
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_complete_batched_instruction_set() {
        let circuit = Circuit::try_parse(
            "I 0\nX 0\nY 0\nZ 0\nS 0\nS_DAG 0\nT 0\nT_DAG 0\n\
             CX 0 1\nCZ 0 1\nSWAP 0 1\nCCZ 0 1 2\nM 0\nMX 1\nMR 0\nMRX 1\nR 0\nRX 1\n\
             X_ERROR(.1) 0\nY_ERROR(.1) 0\nZ_ERROR(.1) 0\nDEPOLARIZE1(.1) 0\n\
             DEPOLARIZE2(.1) 0 1\nDEPOLARIZE3(.1) 0 1 2\nPAULI_CHANNEL_1(.1,.1,.1) 0\n\
             PAULI_CHANNEL_2(.01,.01,.01,.01,.01,.01,.01,.01,.01,.01,.01,.01,.01,.01,.01) 0 1\n\
             MPP(.1) X0*Y1 !Z2\nDETECTOR(1,2) rec[-1] rec[-2]\n\
             OBSERVABLE_INCLUDE(3) rec[-1]\nEXP_VAL X0*Y1 !Z2\n",
        )
        .unwrap();
        assert_eq!(circuit.num_qubits, 3);
        let measurements: usize = circuit
            .instructions
            .iter()
            .filter_map(|instruction| match instruction {
                Instruction::Measure { qubits, .. } => Some(qubits.len()),
                Instruction::MeasurePauliProduct { products, .. } => Some(products.len()),
                _ => None,
            })
            .sum();
        assert_eq!(measurements, 6);
        assert_eq!(circuit.num_detectors(), 1);
        assert_eq!(circuit.num_observables(), 4);
        assert_eq!(circuit.num_exp_vals(), 2);
    }

    #[test]
    fn validates_zero_repeats_and_checked_pairs() {
        assert!(Circuit::try_parse("REPEAT 0 { H 100 }").is_err());
        assert_eq!(
            Circuit::try_parse("REPEAT 0 { X 100 }").unwrap().num_qubits,
            0
        );
        assert!(
            Circuit::try_parse("CX 0")
                .unwrap_err()
                .contains("even number")
        );
        assert!(
            Circuit::try_parse("CX 0 0")
                .unwrap_err()
                .contains("different")
        );
        assert!(
            Circuit::try_parse("CCZ 0 1")
                .unwrap_err()
                .contains("multiple of three")
        );
        assert!(
            Circuit::try_parse("CCZ 0 1 1")
                .unwrap_err()
                .contains("different")
        );
        assert!(
            Circuit::try_parse("DEPOLARIZE3(.1) 0 1")
                .unwrap_err()
                .contains("multiple of three")
        );
    }

    #[test]
    fn rejects_invalid_probabilities_and_products() {
        assert!(
            Circuit::try_parse("X_ERROR(2) 0")
                .unwrap_err()
                .contains("probability")
        );
        assert!(
            Circuit::try_parse("PAULI_CHANNEL_1(.5,.5,.5) 0")
                .unwrap_err()
                .contains("sum")
        );
        for source in [
            "MPP X0*Z0",
            "MPP X0**Z1",
            "MPP A0",
            "MPP X",
            "MPP X-1",
            "MPP(2) X0",
            "EXP_VAL",
            "EXP_VAL X0*Z0",
            "EXP_VAL X0**Z1",
            "EXP_VAL A0",
        ] {
            assert!(Circuit::try_parse(source).is_err(), "accepted {source:?}");
        }
        for source in [
            "DETECTOR rec[-1]",
            "M 0\nDETECTOR rec[-0]",
            "M 0\nDETECTOR rec[1]",
            "M 0\nOBSERVABLE_INCLUDE rec[-1]",
            "M 0\nOBSERVABLE_INCLUDE(-1) rec[-1]",
            "M 0\nOBSERVABLE_INCLUDE(1.5) rec[-1]",
            "OBSERVABLE_INCLUDE(0) X0",
        ] {
            assert!(Circuit::try_parse(source).is_err(), "accepted {source:?}");
        }
    }

    #[test]
    fn parses_and_validates_record_controlled_paulis() {
        let circuit =
            Circuit::try_parse("M 0\nCNOT rec[-1] 4 1 2\nCY rec[-1] 5\nCZ rec[-1] 6\n").unwrap();
        assert_eq!(circuit.num_qubits(), 7);
        assert_eq!(circuit.num_measurements(), 1);
        assert!(matches!(
            circuit.instructions[1],
            Instruction::ClassicallyControlledPauli {
                pauli: Pauli::X,
                lookback: 1,
                target: 4
            }
        ));
        assert!(matches!(
            circuit.instructions[2],
            Instruction::Gate(GateInstruction::Cx(ref pairs)) if pairs == &[(1, 2)]
        ));

        let repeated = Circuit::try_parse("M 0\nREPEAT 2 {\nCX rec[-1] 1\nM 1\n}").unwrap();
        assert_eq!(repeated.num_measurements(), 3);
        assert!(Circuit::try_parse("M 0 1\nCZ rec[-2] 2").is_ok());

        for source in [
            "CX rec[-1] 0",
            "M 0\nCX rec[-0] 1",
            "M 0\nCX !rec[-1] 1",
            "M 0\nCX sweep[0] 1",
            "M 0\nCX 1 rec[-1]",
            "M 0\nCX rec[-1]",
            "CY 0 1",
        ] {
            assert!(Circuit::try_parse(source).is_err(), "accepted {source:?}");
        }
    }

    #[test]
    fn annotations_resolve_records_across_expanded_repeats() {
        let circuit = Circuit::try_parse(
            "M 0\nREPEAT 2 {\nM 0\nDETECTOR rec[-1] rec[-2]\nOBSERVABLE_INCLUDE(2) rec[-1]\n}",
        )
        .unwrap();
        assert_eq!(circuit.num_measurements(), 3);
        assert_eq!(circuit.num_detectors(), 2);
        assert_eq!(circuit.num_observables(), 3);
    }
}
