mod sim;

use sim::Simulation;

fn run_sim() -> Result<(), Box<dyn std::error::Error>> {
    let mut simulation = Simulation::new(0, 1024, 1024, 0.2_f32)?;

    simulation.fill_rect(0, 0, 1024, 1024, 0.0_f32)?;
    simulation.fill_rect(30, 512, 32, 32, 1.0_f32)?;

    simulation.step(1)?;
    simulation.step(1)?;
    simulation.step(1)?;

    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Starting simulation");

    run_sim()?;

    println!("All done.");

    Ok(())
}
