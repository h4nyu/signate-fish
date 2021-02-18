import { range, values, uniq, first, sortBy } from "lodash";

// TODO import charpoints
type Point = {
  x: number;
  y: number;
  imageId: string;
  label?: string;
  confidence?: number;
  isGrandTruth?: boolean;
};

export const Point = (): Point => {
  return {
    x: 0,
    y: 0,
    imageId: "",
    label: undefined,
  };
};

const combinations = (size0: number, size1: number) => {
  const souceIds = range(size0);
  const targetIds = range(size1);
  const res: [number, number][] = [];
  for (const s of souceIds) {
    for (const t of targetIds) {
      res.push([s, t]);
    }
  }
  return uniq(res);
};

type Line = [Point, Point];

export const getBaseline = (points: Point[]): undefined | Line => {
  const preDistance = 0;
  const combPaints = combinations(points.length, points.length)
    .filter((x) => x[0] !== x[1])
    .map((x) => {
      return [points[x[0]], points[x[1]]];
    })
    .map((p) => {
      return [
        p,
        Math.sqrt(Math.pow(p[1].x - p[0].x, 2) + Math.pow(p[1].y - p[0].y, 2)),
      ];
    });

  const line = first(sortBy(combPaints, (x) => -x[1]).map((p) => p[0]));
  return line as any;
};
