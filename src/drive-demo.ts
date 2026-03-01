import { addComponent, addEntity, createWorld, query } from "bitecs";

// --- Types ---

const TripStatus = { ACTIVE: 1, SUCCEEDED: 2, FAILED: 3 } as const;

type SimConfig = {
  fromName: string;
  toName: string;
  fromMiles: number;
  toMiles: number;
  carName: string;
  speedMph: number;
  fuelGallons: number;
  mpg: number;
  tickMinutes?: number;
};

type DriveWorld = ReturnType<typeof createDriveWorld>;

// --- SoA accessor ---

function get(arr: number[], eid: number): number {
  return arr[eid] ?? 0;
}

// --- World setup ---

function createDriveWorld() {
  return createWorld({
    components: {
      CarTag: [] as number[],
      Kinematics: {
        position: [] as number[],
        destination: [] as number[],
        speed: [] as number[],
      },
      Fuel: {
        gallons: [] as number[],
        mpg: [] as number[],
        used: [] as number[],
      },
      Trip: {
        status: [] as number[],
        elapsedMinutes: [] as number[],
      },
    },
  });
}

function setupCar(world: DriveWorld, config: SimConfig): number {
  const { CarTag, Kinematics, Fuel, Trip } = world.components;

  const car = addEntity(world);
  addComponent(world, car, CarTag);
  addComponent(world, car, Kinematics);
  addComponent(world, car, Fuel);
  addComponent(world, car, Trip);

  Kinematics.position[car] = config.fromMiles;
  Kinematics.destination[car] = config.toMiles;
  Kinematics.speed[car] = config.speedMph;
  Fuel.gallons[car] = config.fuelGallons;
  Fuel.mpg[car] = config.mpg;
  Fuel.used[car] = 0;
  Trip.status[car] = TripStatus.ACTIVE;
  Trip.elapsedMinutes[car] = 0;

  return car;
}

// --- Systems ---

function motionSystem(world: DriveWorld, dtHours: number): void {
  const { CarTag, Kinematics, Fuel, Trip } = world.components;

  for (const eid of query(world, [CarTag, Kinematics, Fuel, Trip])) {
    if (get(Trip.status, eid) !== TripStatus.ACTIVE) continue;

    const remaining = get(Kinematics.destination, eid) - get(Kinematics.position, eid);
    const speed = get(Kinematics.speed, eid);
    const mpg = get(Fuel.mpg, eid) || 1;
    const idealTravel = speed * dtHours;
    const maxTravelByFuel = get(Fuel.gallons, eid) * mpg;
    const travel = Math.min(idealTravel, remaining, maxTravelByFuel);
    const fuelBurned = travel / mpg;

    Kinematics.position[eid] = get(Kinematics.position, eid) + travel;
    Fuel.gallons[eid] = get(Fuel.gallons, eid) - fuelBurned;
    Fuel.used[eid] = get(Fuel.used, eid) + fuelBurned;
    Trip.elapsedMinutes[eid] = get(Trip.elapsedMinutes, eid) + dtHours * 60;
  }
}

function constraintSystem(world: DriveWorld): void {
  const { CarTag, Kinematics, Fuel, Trip } = world.components;

  for (const eid of query(world, [CarTag, Kinematics, Fuel, Trip])) {
    if (get(Trip.status, eid) !== TripStatus.ACTIVE) continue;

    const position = get(Kinematics.position, eid);
    const destination = get(Kinematics.destination, eid);

    // Arrival check
    if (position >= destination - 1e-6) {
      Trip.status[eid] = TripStatus.SUCCEEDED;
      continue;
    }

    // Fuel exhaustion check
    if (get(Fuel.gallons, eid) <= 1e-9) {
      Trip.status[eid] = TripStatus.FAILED;
    }
  }
}

// --- Simulation runner ---

function runDriveSimulation(config: SimConfig): void {
  const tickMinutes = config.tickMinutes ?? 1;
  const dtHours = tickMinutes / 60;
  const totalDistance = config.toMiles - config.fromMiles;
  const maxRange = config.fuelGallons * config.mpg;

  const world = createDriveWorld();
  const car = setupCar(world, config);
  const { Kinematics, Fuel, Trip } = world.components;

  console.log(`Simulating drive: ${config.fromName} -> ${config.toName}`);
  console.log(`Car: ${config.carName}`);
  console.log(`Distance: ${totalDistance.toFixed(1)} miles`);
  console.log(`Speed: ${config.speedMph} mph`);
  console.log(`Fuel range: ${maxRange.toFixed(1)} miles`);

  while (get(Trip.status, car) === TripStatus.ACTIVE) {
    motionSystem(world, dtHours);
    constraintSystem(world);
  }

  const status = get(Trip.status, car);

  if (status === TripStatus.FAILED) {
    console.log("Result: FAIL (not enough fuel)");
    console.log(`Distance reached: ${get(Kinematics.position, car).toFixed(2)} miles from start`);
    console.log(`Fuel left: ${get(Fuel.gallons, car).toFixed(2)} gallons`);
    return;
  }

  console.log("Result: PASS");
  console.log(`Estimated duration: ${get(Trip.elapsedMinutes, car).toFixed(1)} minutes`);
  console.log(`Fuel used: ${get(Fuel.used, car).toFixed(2)} gallons`);
  console.log(`Fuel left: ${get(Fuel.gallons, car).toFixed(2)} gallons`);
}

// --- Demo ---

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
