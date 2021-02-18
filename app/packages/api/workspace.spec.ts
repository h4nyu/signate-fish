import { RootApi } from ".";
import { Workspace } from "@sivic/core/workspace";

const rootApi = RootApi();
rootApi.setUrl("http://srv");
describe("spec", () => {
  const api = rootApi.workspace;
  test("create and filter", async () => {
    const savedRow = await api.create({ name: "" });
    if (savedRow instanceof Error) {
      throw savedRow;
    }
    expect(savedRow.name).toBe("");
  });
});
