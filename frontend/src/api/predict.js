import axios from "axios";

const API_BASE = import.meta.env.VITE_API_URL || "http://127.0.0.1:8000";

export async function predictHeartDisease(patientData) {
  const response = await axios.post(`${API_BASE}/predict`, patientData);
  return response.data;
}
