import fastify, { FastifyPlugin } from "fastify";
import { Lock, Store } from "@sivic/core";
import path from "path";
import { WorkspaceRoutes } from "./workspace";
import { ImageRoutes } from "./image";
import { DetectBoxes } from "@sivic/server/store/detect";
import fastifyStatic from "fastify-static";

const urlRoutes: FastifyPlugin<{ prefix: string }> = function (
  app,
  opts,
  done
) {
  app.get("/image-store-url", {}, async (request) => {
    return process.env.IMAGE_STORE_URL;
  });
  done();
};

export const App = (args: { store: Store; lock: Lock }) => {
  const { store, lock } = args;
  const app = fastify({
    bodyLimit: 10048576,
    logger: true,
  });
  const detectBoxes = DetectBoxes(process.env.DETECT_BOXES_URL || "");
  const prefix = path.join("/", process.env.PREFIX || "", "/api/v1");
  app.register(urlRoutes, { prefix });
  app.register(fastifyStatic, {
    root: "/srv/packages/web/dist",
  });
  app.register(WorkspaceRoutes({ store, lock }), {
    prefix: `${prefix}/workspace`,
  });
  app.register(ImageRoutes({ store, lock }), {
    prefix: `${prefix}/image`,
  });
  app.ready(async () => {
    console.log(app.printRoutes());
  });
  return app;
};
