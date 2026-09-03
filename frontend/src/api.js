import axios from "axios";

const client = axios.create({
  baseURL: "http://127.0.0.1:8000",
});

export const getIncidents = (params) =>
  client.get("/api/incidents", { params }).then((r) => r.data);

export const getIncidentStats = () =>
  client.get("/api/incidents/stats").then((r) => r.data);

export const getIncident = (id) =>
  client.get(`/api/incidents/${id}`).then((r) => r.data);

export const investigateIncident = (id) =>
  client.get(`/api/incidents/${id}/investigate`).then((r) => r.data);

export const getTransactions = (params) =>
  client.get("/api/transactions", { params }).then((r) => r.data);

export const getTransaction = (id) =>
  client.get(`/api/transactions/${id}`).then((r) => r.data);

export const scoreTransaction = (payload) =>
  client.post("/api/transactions/score", payload).then((r) => r.data);

export default client;
