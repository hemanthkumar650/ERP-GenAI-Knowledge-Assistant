import { environment } from "./config/environment";
import { createApp } from "./app";

const app = createApp();

app.listen(environment.port, () => {
  console.log(`Backend API listening on http://localhost:${environment.port}`);
});
