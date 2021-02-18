import { Store as _Store } from "@sivic/server/store";

export const Store = () =>
  _Store({
    url: process.env.DATABASE_URL || "",
    imageUrl: process.env.IMAGE_URL || "",
  });
