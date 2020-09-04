import * as yargs from "yargs";
import { AssertionError } from "assert";
import * as fs from "fs";
import * as util from "util";

interface IAxisLine {
  Left: number;
  Right: number;
  N: number;
  Q: number;
  InnerDerive: number;
}

class AxisLine implements IAxisLine {
  constructor(public Left: number, public Right: number, public N = 1, public Q = 1.0, public InnerDerive = 0) {}
}

interface IMapDomain {
  Order: number;
  DomainIndex: number;
  XAxisIndex: number;
  YAxisIndex: number;
  ZAxisIndex: number;
  Function: string;
  LambdaFunction: string;
  GammaFunction: string;
}

interface IMapDomainsData {
  data: Record<string, IMapDomain>;
}

class MapDomainsData implements IMapDomainsData {
  static startingDomainIndex = 1;
  static bFunction = "0";
  static gammaFunction = "0";
  data: Record<string, IMapDomain> = {};
  constructor(input: IInput) {
    const elementsNumber = input.x.elements * input.y.elements * input.z.elements;
    if (input.rhos.length !== elementsNumber) {
      throw new Error(`Number of rho values that is ${input.rhos.length} is not equal to number of elements that is ${elementsNumber}`);
    }
    let currentDomainNumber = MapDomainsData.startingDomainIndex;
    for (let iz = 1; iz < input.z.elements + 1; iz++) {
      for (let iy = 1; iy < input.y.elements + 1; iy++) {
        for (let ix = 1; ix < input.x.elements + 1; ix++) {
          this.data[currentDomainNumber] = {
            Order: 1,
            DomainIndex: currentDomainNumber,
            XAxisIndex: ix,
            YAxisIndex: iy,
            ZAxisIndex: iz,
            Function: MapDomainsData.bFunction,
            LambdaFunction: input.rhos[currentDomainNumber - 1].toString(),
            GammaFunction: MapDomainsData.gammaFunction,
          };
          currentDomainNumber++;
        }
      }
    }
  }
}

interface IBoundaryConditions {
  Surface: "Top" | "Bottom" | "Back" | "Front" | "Left" | "Right";
  XAxisIndex: number;
  YAxisIndex: number;
  ZAxisIndex: number;
  Function: string;
}

interface IAxisLinesData {
  data: Record<string, IAxisLine>;
}

class AxisLinesData implements IAxisLinesData {
  static startingCoordinate = 0;
  static startingIndex = 0;
  data: Record<string, IAxisLine> = {};
  constructor(inputAxis: IInputAxis) {
    let currentCoordinate = AxisLinesData.startingCoordinate;
    const step = inputAxis.boundary / inputAxis.elements;
    for (let i = AxisLinesData.startingIndex; i < inputAxis.elements; i++) {
      const nextCoordinate = currentCoordinate + step;
      this.data[i] = new AxisLine(currentCoordinate, nextCoordinate);
      currentCoordinate = nextCoordinate;
    }
  }
}

export interface IGrid {
  MapXAxisLines: Record<string, IAxisLine>;
  MapYAxisLines: Record<string, IAxisLine>;
  MapZAxisLines: Record<string, IAxisLine>;
  MapDomains: Record<string, IMapDomain>;
  DirichletConditions: IBoundaryConditions[];
  NeumannConditions: IBoundaryConditions[];
}

interface IInputAxis {
  boundary: number;
  elements: number;
}

function assertIsIInputAxis(x: any): asserts x is IInputAxis {
  if (typeof x.boundary === "number" && typeof x.elements === "number" && x.elements > 0) {
    return;
  }
  throw new AssertionError({ message: `Invalid axis: ${JSON.parse(x)}` });
}

interface IInput {
  x: IInputAxis;
  y: IInputAxis;
  z: IInputAxis;
  rhos: number[];
}

function assertIsIInput(x: any): asserts x is IInput {
  assertIsIInputAxis(x.x);
  assertIsIInputAxis(x.y);
  assertIsIInputAxis(x.z);
  if (Array.isArray(x.rhos) && (x.rhos as Array<any>).every(e => typeof e === "number")) {
    return;
  }
  throw new AssertionError({ message: `Invalid input: ${JSON.parse(x)}` });
}

const convert = (input: IInput): IGrid => {
  const grid: IGrid = {
    MapXAxisLines: new AxisLinesData(input.x).data,
    MapYAxisLines: new AxisLinesData(input.y).data,
    MapZAxisLines: new AxisLinesData(input.z).data,
    MapDomains: new MapDomainsData(input).data,
    DirichletConditions: [],
    NeumannConditions: [],
  };
  return grid;
};

(async () => {
  const args = yargs.options({
    inputPath: { type: "string", demandOption: true },
    outputPath: { type: "string", demandOption: true },
  }).argv;
  const input = JSON.parse((await util.promisify(fs.readFile)(args.inputPath)).toString());
  assertIsIInput(input);
  const converted = await convert(input);
  const output = JSON.stringify(converted);
  await util.promisify(fs.writeFile)(args.outputPath, output);
})();
