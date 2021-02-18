import axios from "axios";
import { zip } from "lodash";

export const DetectBoxes = (url: string) => {
  const http = axios.create();
  http.defaults.baseURL = url;
  return async (payload: { data: string }) => {
    try {
      const res = await http.post("/detect", { data: payload.data });
      const { image, boxes, scores } = res.data;
      return zip(boxes, scores).map((c: any) => {
        return {
          x0: c[0][0],
          y0: c[0][1],
          x1: c[0][2],
          y1: c[0][3],
          confidence: c[1],
        };
      });
    } catch (e) {
      return e;
    }
  };
};
