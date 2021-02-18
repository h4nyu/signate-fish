import { Row, Sql } from "postgres";
import { first } from "lodash";

import { ErrorKind } from "@sivic/core";
import { Image } from "@sivic/core/image";
import { ImageStore } from "@sivic/core";
import { RootApi as ImageApi } from "@charpoints/api";

const COLUMNS = [
  "workspace_id",
  "image_id",
  "tag",
  "updated_at",
  "created_at",
] as const;
export const Store = (imageApi: ImageApi, sql: Sql<any>): ImageStore => {
  const to = (r: Row) => {
    return {
      id: r.image_id,
      workspaceId: r.workspace_id,
      tag: r.tag || undefined,
      createdAt: r.created_at,
      updatedAt: r.updated_at,
    };
  };

  const from = (r: Image): Row => {
    return {
      workspace_id: r.workspaceId,
      image_id: r.id,
      tag: r.tag || null,
      created_at: r.createdAt,
      updated_at: r.updatedAt,
    };
  };

  const find = async (payload: { id: string }): Promise<Image | Error> => {
    const image = await imageApi.image.find(payload);
    if (image instanceof Error) {
      return image;
    }
    const rows = await sql`SELECT * FROM workspace_images WHERE image_id = ${image.id} LIMIT 1`;
    const row = first(rows.map(to));
    if (row === undefined) {
      return new Error(ErrorKind.ImageNotFound);
    }
    return {
      ...image,
      ...row,
    };
  };

  const insert = async (payload: Image): Promise<void | Error> => {
    const err = await imageApi.image.create({
      id: payload.id,
      data: payload.data || "",
      name: payload.name,
    });
    if (err instanceof Error) {
      return err;
    }
    try {
      await sql`INSERT INTO workspace_images ${sql(from(payload), ...COLUMNS)}`;
    } catch (e) {
      return e;
    }
  };

  const update = async (payload: Image): Promise<void | Error> => {
    const err = await imageApi.image.update(payload);
    if (err instanceof Error) {
      return err;
    }
    try {
      await sql`UPDATE workspace_images SET ${sql(
        from(payload),
        ...COLUMNS
      )} WHERE image_id = ${payload.id}`;
    } catch (e) {
      return e;
    }
  };

  const delete_ = async (payload: { id: string }): Promise<void | Error> => {
    const err = await imageApi.image.delete(payload);
    if (err instanceof Error) {
      return err;
    }
  };
  return {
    find,
    insert,
    update,
    delete: delete_,
  };
};
