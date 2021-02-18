import { Store } from "@sivic/server";
import fs from "fs";
import { Image, ImageTag } from "@sivic/core/image";

const rootStore = Store();

afterAll(async () => {
  await rootStore.close();
});

describe("image", () => {
  const store = rootStore.image;
  let row = Image();
  beforeAll(async () => {
    const buf = await fs.promises.readFile("/srv/package.json");
    row.data = buf.toString("base64");
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
    expect(res).toEqual(row);
    row = {
      ...row,
      tag: ImageTag.Source,
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
});
