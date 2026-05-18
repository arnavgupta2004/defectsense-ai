import axios from 'axios';

const rawBase = (import.meta.env.VITE_API_BASE_URL ?? '').trim().replace(/\/$/, '');

if (import.meta.env.PROD && !rawBase) {
  console.warn(
    '[DefectSense] VITE_API_BASE_URL is not set. Set it in Vercel → Environment Variables to your FastAPI host (e.g. Render/Railway).',
  );
}

const api = axios.create({
  baseURL: rawBase,
  timeout: 120_000,
});

// -----------------------------
// Backend response types
// -----------------------------

type BackendStatus = 'NORMAL' | 'DEFECTIVE';
type BackendSeverity = 'LOW' | 'MEDIUM' | 'HIGH';

interface BackendDefectRegion {
  bbox: [number, number, number, number];
  severity: BackendSeverity;
  area_percent: number;
}

interface BackendDetectionResult {
  id: number;
  image_id: string;
  filename: string;
  status: BackendStatus;
  anomaly_score: number;
  threshold: number;
  defect_regions: BackendDefectRegion[];
  annotated_image: string;
  inference_time_ms: number;
  timestamp: string;
}

type BackendTrainStatusState = 'IDLE' | 'TRAINING' | 'READY' | 'ERROR';

interface BackendTrainStatus {
  status: BackendTrainStatusState;
  last_trained_at: string | null;
  message?: string | null;
  image_level_auroc?: number | null;
  pixel_level_auroc?: number | null;
  f1_score?: number | null;
  memory_bank_size?: number | null;
  threshold?: number | null;
}

interface BackendDashboardStats {
  total_inspected_today: number;
  pass_rate: number;
  defects_detected: number;
  avg_anomaly_score: number;
  auroc: number;
  defect_distribution: { name: string; value: number; color: string }[];
  recent_results: BackendDetectionResult[];
}

// -----------------------------
// Frontend-facing types
// -----------------------------

export interface DetectionResult {
  image_id: string;
  filename: string;
  category: string;
  status: 'PASS' | 'FAIL';
  anomaly_score: number;
  threshold: number;
  inference_time_ms: number;
  timestamp: string;
  original_image_url: string;
  heatmap_url: string;
  annotated_image_url: string;
  defect_regions: DefectRegion[];
}

export interface DefectRegion {
  id: number;
  bbox: [number, number, number, number];
  severity: 'low' | 'medium' | 'high' | 'critical';
  area_percent: number;
  label: string;
}

export interface ModelStatus {
  status: 'trained' | 'untrained' | 'training' | 'error';
  memory_bank_size: number;
  training_images: number;
  last_trained: string | null;
  message?: string | null;
  metrics: {
    auroc: number;
    f1: number;
    avg_precision: number;
  };
  threshold: number;
}

export interface DashboardStats {
  total_inspected_today: number;
  pass_rate: number;
  defects_detected: number;
  avg_anomaly_score: number;
  auroc: number;
  defect_distribution: { name: string; value: number; color: string }[];
  recent_results: DetectionResult[];
}

// -----------------------------
// Mapping helpers
// -----------------------------

const mapBackendStatus = (status: BackendStatus): DetectionResult['status'] =>
  status === 'NORMAL' ? 'PASS' : 'FAIL';

const mapBackendSeverity = (severity: BackendSeverity): DefectRegion['severity'] => {
  switch (severity) {
    case 'HIGH':
      return 'high';
    case 'MEDIUM':
      return 'medium';
    case 'LOW':
    default:
      return 'low';
  }
};

export const mapBackendDetectionResult = (r: BackendDetectionResult): DetectionResult => ({
  image_id: r.image_id,
  filename: r.filename,
  category: 'Custom',
  status: mapBackendStatus(r.status),
  anomaly_score: r.anomaly_score,
  threshold: r.threshold,
  inference_time_ms: r.inference_time_ms,
  timestamp: r.timestamp,
  original_image_url: '',
  heatmap_url: '',
  annotated_image_url: r.annotated_image ? `data:image/png;base64,${r.annotated_image}` : '',
  defect_regions: r.defect_regions.map((d, idx) => ({
    id: idx + 1,
    bbox: d.bbox,
    severity: mapBackendSeverity(d.severity),
    area_percent: d.area_percent,
    label: `${d.severity} · Region ${idx + 1}`,
  })),
});

const mapBackendTrainStatus = (s: BackendTrainStatus): ModelStatus => {
  let status: ModelStatus['status'] = 'untrained';
  if (s.status === 'READY') status = 'trained';
  else if (s.status === 'TRAINING') status = 'training';
  else if (s.status === 'ERROR') status = 'error';

  return {
    status,
    memory_bank_size: s.memory_bank_size ?? 0,
    training_images: 0,
    last_trained: s.last_trained_at,
    message: s.message ?? undefined,
    metrics: {
      auroc: s.image_level_auroc ?? 0,
      f1: s.f1_score ?? 0,
      avg_precision: 0,
    },
    threshold: s.threshold ?? 0.5,
  };
};

// -----------------------------
// API functions
// -----------------------------

export const uploadImage = async (file: File) => {
  const formData = new FormData();
  formData.append('file', file);
  const { data } = await api.post('/api/upload', formData);
  return data as { image_id: string; filename: string };
};

export const detectDefects = async (imageId: string) => {
  const { data } = await api.post<BackendDetectionResult>(`/api/detect/${imageId}`);
  return mapBackendDetectionResult(data);
};

export const getResult = async (imageId: string) => {
  const { data } = await api.get<BackendDetectionResult>(`/api/results/${imageId}`);
  return mapBackendDetectionResult(data);
};

export const getAllResults = async () => {
  const { data } = await api.get<BackendDetectionResult[]>('/api/results');
  return data.map(mapBackendDetectionResult);
};

export const trainModel = async (datasetPath?: string) => {
  const { data } = await api.post<BackendTrainStatus>('/api/train', {
    dataset_path: datasetPath ?? null,
  });
  return mapBackendTrainStatus(data);
};

export const getModelStatus = async () => {
  const { data } = await api.get<BackendTrainStatus>('/api/model/status');
  return mapBackendTrainStatus(data);
};

export const updateThreshold = async (threshold: number) => {
  const { data } = await api.patch<BackendTrainStatus>('/api/model/threshold', { threshold });
  return mapBackendTrainStatus(data);
};

export const getDashboardStats = async (): Promise<DashboardStats> => {
  const { data } = await api.get<BackendDashboardStats>('/api/dashboard');
  return {
    ...data,
    recent_results: data.recent_results.map(mapBackendDetectionResult),
  };
};

export default api;
