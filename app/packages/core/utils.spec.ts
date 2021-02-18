import { getBaseline, Point } from "@sivic/core/utils";

describe("spec", () => {
  test("create and filter", async () => {
    const points = [
      { ...Point(), x: 0, y: 0 },
      { ...Point(), x: 2, y: 2 },
      { ...Point(), x: 5, y: 5 },
    ];
    const baseline = getBaseline(points);
    console.log(baseline);
    expect(baseline).toEqual([
      { ...Point(), x: 0, y: 0 },
      { ...Point(), x: 5, y: 5 },
    ]);
  });
});
