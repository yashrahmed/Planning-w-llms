import { addComponent, addEntity, createWorld, query } from "bitecs";

type SimConfig = {
  fromName: string;
  toName: string;
  fromMiles: number;
  toMiles: number;
  carName: string;
  speedMph: number;
  fuelGallons: number;
  mpg: number;
};

type DriveWorld = ReturnType<typeof createDriveWorld>;

function createDriveWorld() {
  return createWorld({
    components: {
      CarTag: [] as number[],
      Position: { miles: [] as number[] },
      Destination: { miles: [] as number[] },
      Motion: { speedMph: [] as number[] },
      Fuel: {
        gallons: [] as number[],
        mpg: [] as number[],
        usedGallons: [] as number[],
      },
      TripState: {
        active: [] as number[],
        succeeded: [] as number[],
        failed: [] as number[],
        elapsedMinutes: [] as number[],
      },
    },
  });
}

const dtHours = 1 / 60; // 1-minute ticks

function setupDrive(world: DriveWorld, config: SimConfig): number {
  const {
    CarTag,
    Position,
    Destination,
    Motion,
    Fuel,
    TripState,
  } = world.components;

  const car = addEntity(world);
  addComponent(world, car, CarTag);
  addComponent(world, car, Position);
  addComponent(world, car, Destination);
  addComponent(world, car, Motion);
  addComponent(world, car, Fuel);
  addComponent(world, car, TripState);

  Position.miles[car] = config.fromMiles;
  Destination.miles[car] = config.toMiles;
  Motion.speedMph[car] = config.speedMph;
  Fuel.gallons[car] = config.fuelGallons;
  Fuel.mpg[car] = config.mpg;
  Fuel.usedGallons[car] = 0;
  TripState.active[car] = 1;
  TripState.succeeded[car] = 0;
  TripState.failed[car] = 0;
  TripState.elapsedMinutes[car] = 0;

  return car;
}

function stepDriveSystem(world: DriveWorld): void {
  const {
    CarTag,
    Position,
    Destination,
    Motion,
    Fuel,
    TripState,
  } = world.components;

  for (const eid of query(world, [CarTag, Position, Destination, Motion, Fuel, TripState])) {
    if (!(TripState.active[eid] ?? 0)) {
      continue;
    }

    const current = Position.miles[eid] ?? 0;
    const target = Destination.miles[eid] ?? 0;
    const remaining = target - current;

    if (remaining <= 0) {
      TripState.active[eid] = 0;
      TripState.succeeded[eid] = 1;
      continue;
    }

    const speed = Motion.speedMph[eid] ?? 0;
    const fuelGallons = Fuel.gallons[eid] ?? 0;
    const mpg = Fuel.mpg[eid] ?? 1;
    const idealTravel = speed * dtHours;
    const maxTravelByFuel = fuelGallons * mpg;
    const travel = Math.min(idealTravel, remaining, maxTravelByFuel);
    const fuelUsed = travel / mpg;

    Position.miles[eid] = (Position.miles[eid] ?? 0) + travel;
    Fuel.gallons[eid] = (Fuel.gallons[eid] ?? 0) - fuelUsed;
    Fuel.usedGallons[eid] = (Fuel.usedGallons[eid] ?? 0) + fuelUsed;
    TripState.elapsedMinutes[eid] = (TripState.elapsedMinutes[eid] ?? 0) + dtHours * 60;

    if (Position.miles[eid] >= target - 1e-6) {
      TripState.active[eid] = 0;
      TripState.succeeded[eid] = 1;
      continue;
    }

    if (travel < idealTravel) {
      TripState.active[eid] = 0;
      TripState.failed[eid] = 1;
    }
  }
}

function runDriveSimulation(config: SimConfig): void {
  const totalDistance = config.toMiles - config.fromMiles;
  const maxRange = config.fuelGallons * config.mpg;
  const world = createDriveWorld();
  const car = setupDrive(world, config);
  const { Position, Fuel, TripState } = world.components;

  console.log(`Simulating drive: ${config.fromName} -> ${config.toName}`);
  console.log(`Car: ${config.carName}`);
  console.log(`Distance: ${totalDistance.toFixed(1)} miles`);
  console.log(`Speed: ${config.speedMph} mph`);
  console.log(`Fuel range: ${maxRange.toFixed(1)} miles`);

  while ((TripState.active[car] ?? 0) === 1) {
    stepDriveSystem(world);
  }

  if ((TripState.failed[car] ?? 0) === 1) {
    console.log("Result: FAIL (not enough fuel)");
    console.log(`Distance reached: ${(Position.miles[car] ?? 0).toFixed(2)} miles from start`);
    console.log(`Fuel left: ${(Fuel.gallons[car] ?? 0).toFixed(2)} gallons`);
    return;
  }

  console.log("Result: PASS");
  console.log(`Estimated duration: ${(TripState.elapsedMinutes[car] ?? 0).toFixed(1)} minutes`);
  console.log(`Fuel used: ${(Fuel.usedGallons[car] ?? 0).toFixed(2)} gallons`);
  console.log(`Fuel left: ${(Fuel.gallons[car] ?? 0).toFixed(2)} gallons`);
}

runDriveSimulation({
  fromName: "Point A",
  toName: "Point B",
  fromMiles: 0,
  toMiles: 30,
  carName: "Demo Car",
  speedMph: 60,
  fuelGallons: 2,
  mpg: 20,
});
