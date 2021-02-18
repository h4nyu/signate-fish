import { Store } from "@sivic/server";
import fs from "fs";
import { Workspace } from "@sivic/core/workspace";

const rootStore = Store();

afterAll(async () => {
  await rootStore.close();
});

describe("workspace", () => {
  const store = rootStore.workspace;
  let row = {
    ...Workspace(),
    imageIds: ["a", "b"],
  };

  beforeAll(async () => {
    await store.clear();
  });

  test("insert, find and update", async () => {
    const err = await store.insert(row);
    if (err instanceof Error) {
      throw err;
    }
    let res = await store.find({ id: row.id });
    if (res instanceof Error) {
      throw res;
    }
    row = {
      ...row,
      name: "update",
    };
    const update = await store.update(row);
    if (update instanceof Error) {
      throw update;
    }
    res = await store.find({ id: row.id });
    if (res instanceof Error) {
      throw res;
    }
    expect(res).toEqual(row);
  });
  test("filter", async () => {
    const rows = await store.filter({});
    if (rows instanceof Error) {
      throw rows;
    }
    expect(rows).toMatchObject([row]);
  });
  test("delete", async () => {
    const err = await store.delete({ id: row.id });
    if (err instanceof Error) {
      throw err;
    }
    const rows = await store.filter({});
    if (rows instanceof Error) {
      throw rows;
    }
    expect(rows).toEqual([]);
  });
});
