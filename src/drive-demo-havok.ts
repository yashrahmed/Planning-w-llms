/// <reference lib="dom" />

import HavokPhysics from "@babylonjs/havok";
import { FreeCamera, MeshBuilder, NullEngine, Scene, Vector3 } from "@babylonjs/core";
import { HavokPlugin, PhysicsAggregate, PhysicsShapeType } from "@babylonjs/core/Physics/v2/index.js";
import "@babylonjs/core/Physics/v2/physicsEngineComponent.js";

type SimConfig = {
  fromName: string;
  toName: string;
  distanceMiles: number;
  speedMph: number;
  fuelGallons: number;
  mpg: number;
};

const METERS_PER_MILE = 1609.34;
const SECONDS_PER_HOUR = 3600;
const DT_SECONDS = 10 / 60; // 60Hz

async function runHavokDriveDemo(config: SimConfig): Promise<void> {
  const hk = await HavokPhysics();
  const engine = new NullEngine({
    renderWidth: 1,
    renderHeight: 1,
    textureSize: 1,
    deterministicLockstep: true,
    lockstepMaxSteps: 1,
  });
  const scene = new Scene(engine);
  const plugin = new HavokPlugin(true, hk);
  scene.enablePhysics(new Vector3(0, 0, 0), plugin);
  scene.getPhysicsEngine()?.setTimeStep(DT_SECONDS);
  const camera = new FreeCamera("camera", new Vector3(0, 10, -20), scene);
  camera.setTarget(Vector3.Zero());
  scene.activeCamera = camera;

  const ground = MeshBuilder.CreateGround("ground", { width: 20, height: 20 }, scene);
  new PhysicsAggregate(ground, PhysicsShapeType.BOX, { mass: 0 }, scene);

  const car = MeshBuilder.CreateBox("car", { width: 2, height: 1, depth: 4 }, scene);
  car.position = new Vector3(0, 0.6, 0);
  const carAggregate = new PhysicsAggregate(
    car,
    PhysicsShapeType.BOX,
    { mass: 1200, friction: 0.5, restitution: 0.0 },
    scene,
  );

  const distanceMeters = config.distanceMiles * METERS_PER_MILE;
  const speedMps = (config.speedMph * METERS_PER_MILE) / SECONDS_PER_HOUR;
  const maxRangeMiles = config.fuelGallons * config.mpg;
  const maxTravelMeters = maxRangeMiles * METERS_PER_MILE;

  let traveledMeters = 0;
  let fuelLeftGallons = config.fuelGallons;
  let tickCount = 0;
  let failed = false;

  // Constant velocity "drive" for demo purposes.
  carAggregate.body.setLinearVelocity(new Vector3(speedMps, 0, 0));

  console.log(`Havok drive simulation: ${config.fromName} -> ${config.toName}`);
  console.log(`Distance: ${config.distanceMiles.toFixed(1)} miles`);
  console.log(`Speed: ${config.speedMph} mph`);
  console.log(`Fuel range: ${maxRangeMiles.toFixed(1)} miles`);

  while (traveledMeters < distanceMeters) {
    scene.render();
    tickCount += 1;

    const stepTravel = speedMps * DT_SECONDS;
    traveledMeters += stepTravel;
    const stepFuelUsed = (stepTravel / METERS_PER_MILE) / config.mpg;
    fuelLeftGallons -= stepFuelUsed;

    if (traveledMeters > maxTravelMeters || fuelLeftGallons < 0) {
      failed = true;
      break;
    }
  }

  const elapsedMinutes = (tickCount * DT_SECONDS) / 60;
  const traveledMiles = traveledMeters / METERS_PER_MILE;

  if (failed) {
    console.log("Result: FAIL (not enough fuel)");
    console.log(`Ticks: ${tickCount}`);
    console.log(`Distance reached: ${traveledMiles.toFixed(2)} miles`);
    console.log(`Fuel left: ${Math.max(fuelLeftGallons, 0).toFixed(2)} gallons`);
  } else {
    console.log("Result: PASS");
    console.log(`Ticks: ${tickCount}`);
    console.log(`Estimated duration: ${elapsedMinutes.toFixed(1)} minutes`);
    console.log(`Fuel used: ${(config.fuelGallons - fuelLeftGallons).toFixed(2)} gallons`);
    console.log(`Fuel left: ${fuelLeftGallons.toFixed(2)} gallons`);
  }

  scene.dispose();
  engine.dispose();
}

await runHavokDriveDemo({
  fromName: "Point A",
  toName: "Point B",
  distanceMiles: 30,
  speedMph: 60,
  fuelGallons: 2,
  mpg: 20,
});
