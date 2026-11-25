mod sim;

use std::collections;
use std::fs::File;
use std::io::Write;

use sim::Field;
use sim::Point;
use sim::Simulation;

use crate::sim::SimError;

fn sample(x: usize, y: usize, field: &Field) -> Result<f32, SimError> {
    field.get(Point { x, y })
}

fn check_sum(field: &Field, goal: f64, threshold: f64) {
    let total: f64 = field.sum();
    assert!(
        (total - goal).abs() < threshold,
        "sum {total} not within {threshold} of {goal}"
    );
}

fn write_to_disk_pgm(field: &Field) -> Result<(), Box<dyn std::error::Error>> {
    let bytes: Vec<u8> = field.data.iter().map(|f| (f * 255_f32) as u8).collect();

    let path = std::env::current_dir()?.join("out.pgm");
    let mut file = File::create(&path)?;

    write!(file, "P5\n{} {}\n255\n", field.width, field.height)?;
    file.write_all(&bytes)?;

    Ok(())
}

fn run_sim() -> Result<(), Box<dyn std::error::Error>> {
    let target_sum = 1024_f64;
    let target_epsilon = 1e-3_f64;
    let mut simulation = Simulation::new(0, 1024, 1024, 0.2_f32)?;

    simulation.fill_rect(0, 0, simulation.width(), simulation.height(), 0.0_f32)?;
    simulation.fill_rect(30, 512, 32, 32, 1.0_f32)?;

    let field = simulation.download()?;
    println!("Spot check: {}", sample(42, 512, &field)?);
    check_sum(&field, target_sum, target_epsilon);

    simulation.step(10)?;
    let field = simulation.download()?;
    println!("Spot check: {}", sample(42, 512, &field)?);

    simulation.step(10)?;
    let field = simulation.download()?;
    println!("Spot check: {}", sample(42, 512, &field)?);

    simulation.step(10)?;
    let field = simulation.download()?;
    println!("Spot check: {}", sample(42, 512, &field)?);

    simulation.step(100)?;
    let field = simulation.download()?;
    check_sum(&field, target_sum, target_epsilon);

    write_to_disk_pgm(&field)?;

    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Starting simulation");

    run_sim()?;

    println!("All done.");

    Ok(())
}
