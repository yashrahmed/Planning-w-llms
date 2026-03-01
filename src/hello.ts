type LocationType = "place" | "gas_station";

type LegResult = {
  from: string;
  to: string;
  distanceMiles: number;
  fuelBefore: number;
  fuelAfter: number;
  refueledAtStart: boolean;
};

type ValidationResult = {
  ok: boolean;
  failureReason?: string;
  legs: LegResult[];
};

class Location {
  public readonly id: string;
  public readonly name: string;
  public readonly type: LocationType;

  public constructor(id: string, name: string, type: LocationType) {
    if (!id.trim()) {
      throw new Error("Location.id must be non-empty.");
    }
    if (!name.trim()) {
      throw new Error("Location.name must be non-empty.");
    }

    this.id = id;
    this.name = name;
    this.type = type;
  }
}

class VehicleL2 {
  public readonly make: string;
  public readonly model: string;
  public readonly fuelRangeMiles: number;

  public constructor(make: string, model: string, fuelRangeMiles: number) {
    if (fuelRangeMiles <= 0) {
      throw new Error("VehicleL2.fuelRangeMiles must be > 0.");
    }

    this.make = make;
    this.model = model;
    this.fuelRangeMiles = fuelRangeMiles;
  }
}

class DriveL1 {
  public readonly driver: string;
  public readonly passengers: string[];
  public readonly vehicle: string;
  public readonly startLocation: string;
  public readonly destination: string;
  public readonly waypoints: string[];

  public constructor(args: {
    driver: string;
    passengers: string[];
    vehicle: string;
    startLocation: string;
    destination: string;
    waypoints: string[];
  }) {
    this.driver = args.driver;
    this.passengers = args.passengers;
    this.vehicle = args.vehicle;
    this.startLocation = args.startLocation;
    this.destination = args.destination;
    this.waypoints = args.waypoints;
  }
}

class LocationResolver {
  private readonly locationsByName: Map<string, Location>;
  private readonly distances: Map<string, Map<string, number>>;

  public constructor() {
    const locations = [
      new Location("home", "Home", "place"),
      new Location("waypoint-a", "Waypoint A", "place"),
      new Location("waypoint-b", "Waypoint B", "place"),
      new Location("campground", "Campground", "place"),
      new Location("gas-a", "Gas Station A", "gas_station"),
      new Location("gas-b", "Gas Station B", "gas_station"),
    ];

    this.locationsByName = new Map(locations.map((loc) => [loc.name, loc]));
    this.distances = new Map();

    this.addSymmetricDistance("Home", "Waypoint A", 120);
    this.addSymmetricDistance("Waypoint A", "Waypoint B", 70);
    this.addSymmetricDistance("Waypoint B", "Campground", 50);
    this.addSymmetricDistance("Waypoint A", "Gas Station A", 0);
    this.addSymmetricDistance("Gas Station A", "Waypoint B", 70);
    this.addSymmetricDistance("Waypoint B", "Gas Station B", 4);
    this.addSymmetricDistance("Gas Station B", "Campground", 86);
  }

  public resolveLocation(name: string): Location {
    const location = this.locationsByName.get(name);
    if (!location) {
      throw new Error(`Unknown location: "${name}"`);
    }
    return location;
  }

  public resolveVehicle(name: string): VehicleL2 {
    if (name === "2016 Toyota Camry") {
      return new VehicleL2("Toyota", "Camry 2016", 120);
    }
    throw new Error(`Unknown vehicle: "${name}"`);
  }

  public distance(a: Location, b: Location): number {
    const row = this.distances.get(a.name);
    const distance = row?.get(b.name);

    if (distance === undefined) {
      throw new Error(`Missing distance for leg "${a.name}" -> "${b.name}"`);
    }

    return distance;
  }

  private addSymmetricDistance(a: string, b: string, miles: number): void {
    this.addDirectedDistance(a, b, miles);
    this.addDirectedDistance(b, a, miles);
  }

  private addDirectedDistance(from: string, to: string, miles: number): void {
    const row = this.distances.get(from) ?? new Map<string, number>();
    row.set(to, miles);
    this.distances.set(from, row);
  }
}

class DriveL2 {
  public readonly driver: string;
  public readonly passengers: string[];
  public readonly vehicle: VehicleL2;
  public readonly startLocation: Location;
  public readonly destination: Location;
  public readonly waypoints: Location[];

  public constructor(args: {
    driver: string;
    passengers: string[];
    vehicle: VehicleL2;
    startLocation: Location;
    destination: Location;
    waypoints: Location[];
  }) {
    this.driver = args.driver;
    this.passengers = args.passengers;
    this.vehicle = args.vehicle;
    this.startLocation = args.startLocation;
    this.destination = args.destination;
    this.waypoints = args.waypoints;
  }

  public static fromL1(frame: DriveL1, resolver: LocationResolver): DriveL2 {
    return new DriveL2({
      driver: frame.driver,
      passengers: frame.passengers,
      vehicle: resolver.resolveVehicle(frame.vehicle),
      startLocation: resolver.resolveLocation(frame.startLocation),
      destination: resolver.resolveLocation(frame.destination),
      waypoints: frame.waypoints.map((name) => resolver.resolveLocation(name)),
    });
  }

  public route(): Location[] {
    return [this.startLocation, ...this.waypoints, this.destination];
  }

  public validateFuelFeasibility(resolver: LocationResolver): ValidationResult {
    const route = this.route();
    let fuelRemaining = this.vehicle.fuelRangeMiles;
    const legs: LegResult[] = [];

    for (let i = 0; i < route.length - 1; i += 1) {
      const from = route[i];
      const to = route[i + 1];
      if (!from || !to) {
        throw new Error("Invalid route construction.");
      }

      const refueledAtStart = from.type === "gas_station";
      if (refueledAtStart) {
        fuelRemaining = this.vehicle.fuelRangeMiles;
      }

      const fuelBefore = fuelRemaining;
      const distanceMiles = resolver.distance(from, to);
      fuelRemaining -= distanceMiles;

      legs.push({
        from: from.name,
        to: to.name,
        distanceMiles,
        fuelBefore,
        fuelAfter: fuelRemaining,
        refueledAtStart,
      });

      if (fuelRemaining < 0) {
        return {
          ok: false,
          failureReason: `Ran out of fuel on leg ${from.name} -> ${to.name}. Needed ${distanceMiles} miles but had ${fuelBefore}.`,
          legs,
        };
      }
    }

    return { ok: true, legs };
  }
}

function printScenario(
  title: string,
  frameL1: DriveL1,
  frameL2: DriveL2,
  result: ValidationResult,
): void {
  const status = result.ok ? "PASS" : "FAIL";
  console.log(`\n=== ${title}: ${status} ===`);
  console.log(`L1 vehicle: ${frameL1.vehicle}`);
  console.log(
    `L2 route: ${frameL2.route().map((location) => location.name).join(" -> ")}`,
  );

  if (!result.ok) {
    console.log(`Failure: ${result.failureReason}`);
  } else {
    console.log("Validation succeeded without running out of fuel.");
  }

  console.log("Leg trace:");
  for (const leg of result.legs) {
    console.log(
      `- ${leg.from} -> ${leg.to} | distance=${leg.distanceMiles} | fuel_before=${leg.fuelBefore} | fuel_after=${leg.fuelAfter} | refueled_at_start=${leg.refueledAtStart}`,
    );
  }
}

function main(): void {
  const resolver = new LocationResolver();

  const noGasStopsL1 = new DriveL1({
    driver: "Alex",
    passengers: ["Sam", "Dana"],
    vehicle: "2016 Toyota Camry",
    startLocation: "Home",
    destination: "Campground",
    waypoints: ["Waypoint A", "Waypoint B"],
  });

  const noGasStopsL2 = DriveL2.fromL1(noGasStopsL1, resolver);
  const noGasStopsValidation = noGasStopsL2.validateFuelFeasibility(resolver);
  printScenario(
    "Scenario 1 (No gas stops)",
    noGasStopsL1,
    noGasStopsL2,
    noGasStopsValidation,
  );

  const withGasStopsL1 = new DriveL1({
    driver: "Alex",
    passengers: ["Sam", "Dana"],
    vehicle: "2016 Toyota Camry",
    startLocation: "Home",
    destination: "Campground",
    waypoints: ["Waypoint A", "Gas Station A", "Waypoint B"],
  });

  const withGasStopsL2 = DriveL2.fromL1(withGasStopsL1, resolver);
  const withGasStopsValidation = withGasStopsL2.validateFuelFeasibility(resolver);
  printScenario(
    "Scenario 2 (Gas stops inserted)",
    withGasStopsL1,
    withGasStopsL2,
    withGasStopsValidation,
  );
}

main();
