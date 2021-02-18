import postgres from "postgres";
import { Store as WorkspaceStore } from "./workspace";
import { Store as ImageStore } from "./image";
import { RootApi as ImageApi } from "@charpoints/api";

export const Store = (args: {
  url: string;
  max?: number;
  imageUrl: string;
}) => {
  const sql = postgres(args.url, {
    max: args.max || 5,
  });
  const imageApi = ImageApi();
  imageApi.setUrl(args.imageUrl);
  const close = async () => {
    await sql.end({ timeout: 5 });
  };
  const workspace = WorkspaceStore(sql);
  const image = ImageStore(imageApi, sql);
  return {
    workspace,
    image,
    close,
  };
};
export default Store;
