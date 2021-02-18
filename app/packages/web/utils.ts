export function b64toBlob(base64: string): Blob | Error {
  const bin = atob(base64.replace(/^.*,/, ""));
  const buffer = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) {
    buffer[i] = bin.charCodeAt(i);
  }
  try {
    return new Blob([buffer.buffer]);
  } catch (e) {
    return e;
  }
}
export const readAsBase64 = (file: File): Promise<string | Error> =>
  new Promise((resolve) => {
    const reader = new FileReader();
    reader.readAsDataURL(file);
    reader.onload = () => {
      const result = reader.result;
      if (typeof result === "string") {
        resolve(result.split(",")[1]);
      } else if (result instanceof ArrayBuffer) {
        resolve(new Error("ArrayBuffer is Unsupported"));
      } else {
        resolve(new Error("Fail to convert file to base64"));
      }
    };
    reader.onerror = () => resolve(new Error("Fail to convert file to base64"));
  });

export const readAsText = (file: File): Promise<string | Error> =>
  new Promise((resolve) => {
    const reader = new FileReader();
    reader.readAsText(file);
    reader.onload = () => {
      const result = reader.result;
      if (typeof result === "string") {
        resolve(result);
      } else if (result instanceof ArrayBuffer) {
        resolve(new Error("ArrayBuffer is Unsupported"));
      } else {
        resolve(new Error("Fail to convert file to base64"));
      }
    };
    reader.onerror = () => resolve(new Error("Fail to convert file to base64"));
  });
