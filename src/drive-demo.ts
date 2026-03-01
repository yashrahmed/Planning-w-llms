import RAPIER from "@dimforge/rapier3d-compat";

type SimConfig = {
  fromName: string;
  toName: string;
  distanceMiles: number;
  carName: string;
  speedMph: number;
  fuelGallons: number;
  mpg: number;
  dtSeconds?: number;
};

const METERS_PER_MILE = 1609.34;

async function runDriveSimulation(config: SimConfig): Promise<void> {
  await RAPIER.init();

  const dtSeconds = config.dtSeconds ?? 5 / 60;
  const speedMps = (config.speedMph * METERS_PER_MILE) / 3600;
  const distanceMeters = config.distanceMiles * METERS_PER_MILE;
  const maxRangeMiles = config.fuelGallons * config.mpg;

  const world = new RAPIER.World({ x: 0, y: 0, z: 0 });
  world.timestep = dtSeconds;

  // Static road plane.
  const groundBody = world.createRigidBody(RAPIER.RigidBodyDesc.fixed().setTranslation(0, -0.5, 0));
  world.createCollider(RAPIER.ColliderDesc.cuboid(2000, 0.5, 10), groundBody);

  // Dynamic car body.
  const carBody = world.createRigidBody(RAPIER.RigidBodyDesc.dynamic().setTranslation(0, 0.5, 0));
  world.createCollider(RAPIER.ColliderDesc.cuboid(1, 0.5, 2), carBody);
  carBody.setLinvel({ x: speedMps, y: 0, z: 0 }, true);

  let ticks = 0;
  let traveledMeters = 0;
  let lastX = carBody.translation().x;
  let fuelLeftGallons = config.fuelGallons;
  let failed = false;

  console.log(`Simulating drive (Rapier): ${config.fromName} -> ${config.toName}`);
  console.log(`Car: ${config.carName}`);
  console.log(`Distance: ${config.distanceMiles.toFixed(1)} miles`);
  console.log(`Speed: ${config.speedMph} mph`);
  console.log(`Fuel range: ${maxRangeMiles.toFixed(1)} miles`);

  while (traveledMeters < distanceMeters) {
    world.step();
    ticks += 1;

    const currentX = carBody.translation().x;
    const stepTravelMeters = Math.max(0, currentX - lastX);
    lastX = currentX;
    traveledMeters += stepTravelMeters;

    const stepTravelMiles = stepTravelMeters / METERS_PER_MILE;
    fuelLeftGallons -= stepTravelMiles / config.mpg;

    if (fuelLeftGallons < 0) {
      failed = true;
      break;
    }
  }

  const elapsedMinutes = (ticks * dtSeconds) / 60;
  const traveledMiles = traveledMeters / METERS_PER_MILE;

  if (failed) {
    console.log("Result: FAIL (not enough fuel)");
    console.log(`Ticks: ${ticks}`);
    console.log(`Distance reached: ${traveledMiles.toFixed(2)} miles`);
    console.log(`Fuel left: ${Math.max(0, fuelLeftGallons).toFixed(2)} gallons`);
    return;
  }

  console.log("Result: PASS");
  console.log(`Ticks: ${ticks}`);
  console.log(`Estimated duration: ${elapsedMinutes.toFixed(1)} minutes`);
  console.log(`Fuel used: ${(config.fuelGallons - fuelLeftGallons).toFixed(2)} gallons`);
  console.log(`Fuel left: ${fuelLeftGallons.toFixed(2)} gallons`);
}

await runDriveSimulation({
  fromName: "Point A",
  toName: "Point B",
  distanceMiles: 30,
  carName: "Demo Car",
  speedMph: 60,
  fuelGallons: 2,
  mpg: 20,
});
