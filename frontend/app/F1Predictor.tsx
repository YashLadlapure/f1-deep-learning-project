"use client";
import { useState } from 'react';

type PredictorFormData = {
  lap: number;
  position: number;
  pit_stop: number;
  tyre_age: number;
  grid: number;
  alt: number;
  driver_skill: number;
  circuit_difficulty: number;
  race_year: number;
  round: number;
  prev_lap_time: number;
};

const INPUTS: Array<{
  name: keyof PredictorFormData;
  label: string;
  helper: string;
  min?: number;
  step?: number;
  placeholder: string;
}> = [
  {
    name: 'lap',
    label: 'Lap number',
    helper: 'Current lap in the race.',
    min: 1,
    step: 1,
    placeholder: '35',
  },
  {
    name: 'position',
    label: 'Current Position',
    helper: 'Current position in the race.',
    min: 1,
    step: 1,
    placeholder: '4',
  },
  {
    name: 'pit_stop',
    label: 'Pit stops',
    helper: 'Total stops made so far.',
    min: 0,
    step: 1,
    placeholder: '0',
  },
  {
    name: 'tyre_age',
    label: 'Tyre age',
    helper: 'Tyre age in laps.',
    min: 0,
    step: 1,
    placeholder: '21',
  },
  {
    name: 'grid',
    label: 'Grid position',
    helper: 'Starting position on the grid.',
    min: 1,
    step: 1,
    placeholder: '4',
  },
  {
    name: 'alt',
    label: 'Track Altitude',
    helper: 'Altitude of the circuit.',
    step: 1,
    placeholder: '7',
  },
  {
    name: 'driver_skill',
    label: 'Driver Skill Rating',
    helper: 'Calculated metric for the driver.',
    step: 0.01, // Allowing decimals for the skill metric
    placeholder: '92836.56',
  },
  {
    name: 'circuit_difficulty',
    label: 'Circuit Difficulty Rating',
    helper: 'Calculated metric for the track.',
    step: 0.01, // Allowing decimals
    placeholder: '100465.74',
  },
  {
    name: 'race_year',
    label: 'Race year',
    helper: 'Season used by the model.',
    min: 1950,
    step: 1,
    placeholder: '2010',
  },
  {
    name: 'round',
    label: 'Championship Round',
    helper: 'Round number in the season.',
    min: 1,
    step: 1,
    placeholder: '1',
  },
  {
    name: 'prev_lap_time',
    label: 'Previous Lap Time (ms)',
    helper: 'Time taken for the previous lap in milliseconds.',
    step: 1,
    placeholder: '119706.0',
  },
];

export default function F1Predictor() {

  const [formData, setFormData] = useState<PredictorFormData>({
    lap: 35,
    position: 4,
    pit_stop: 0,
    tyre_age: 21,
    grid: 4.0,
    alt: 7,
    driver_skill: 92836.56,
    circuit_difficulty: 100465.74,
    race_year: 2010,
    round: 1,
    prev_lap_time: 119706.0
  });

  const [prediction, setPrediction] = useState<number | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();

    setIsLoading(true);
    setError(null);

    try {
      const res = await fetch('http://localhost:5000/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(formData)
      });

      const data = await res.json();

      if (!res.ok) {
        throw new Error(data.message || 'Prediction request failed');
      }

      setPrediction(data.predicted_lap_time_ms);
    } catch (err) {
      setPrediction(null);
      setError(err instanceof Error ? err.message : 'Unable to generate prediction');
    } finally {
      setIsLoading(false);
    }
  };

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const { name, value } = e.target;
    setFormData({ ...formData, [name]: Number(value) } as PredictorFormData);
  };

  return (
    <main className="min-h-screen bg-slate-100 px-4 py-10 text-slate-900 sm:px-6 lg:px-8">
      <div className="mx-auto w-full max-w-3xl">
        <div className="rounded-3xl border border-slate-200 bg-white p-6 shadow-sm sm:p-8">
          <div className="mb-6">
            <p className="text-sm font-medium uppercase tracking-[0.24em] text-slate-500">
              F1 Lap Time Predictor
            </p>
            <h1 className="mt-2 text-3xl font-semibold tracking-tight text-slate-900">
              Predict a lap time
            </h1>
            <p className="mt-2 text-sm leading-6 text-slate-600">
              Enter the race details below and get a lap time estimate.
            </p>
          </div>

          <form onSubmit={handleSubmit} className="space-y-4">
            <div className="grid gap-4 sm:grid-cols-2">
              {INPUTS.map((field) => (
                <label key={field.name} className="space-y-2">
                  <span className="block text-sm font-medium text-slate-700">{field.label}</span>
                  <input
                    type="number"
                    name={field.name}
                    min={field.min}
                    step={field.step}
                    value={formData[field.name]}
                    onChange={handleChange}
                    className="w-full rounded-xl border border-slate-300 bg-white px-3 py-2 text-slate-900 outline-none transition focus:border-slate-400 focus:ring-2 focus:ring-slate-200"
                  />
                  <span className="block text-xs text-slate-500">{field.helper}</span>
                </label>
              ))}
            </div>

            <button
              type="submit"
              disabled={isLoading}
              className="inline-flex w-full items-center justify-center rounded-xl bg-slate-900 px-4 py-3 text-sm font-semibold text-white transition hover:bg-slate-800 disabled:cursor-not-allowed disabled:opacity-70"
            >
              {isLoading ? 'Predicting...' : 'Predict lap time'}
            </button>
          </form>

          <div className="mt-6 rounded-2xl border border-slate-200 bg-slate-50 p-5" aria-live="polite">
            <p className="text-xs font-medium uppercase tracking-[0.24em] text-slate-500">
              Result
            </p>
            {prediction !== null ? (
              <div className="mt-3">
                <p className="text-3xl font-semibold text-slate-900">
                  {(prediction / 1000).toFixed(3)} seconds
                </p>
                <p className="mt-1 text-sm text-slate-600">
                  Raw value: {prediction.toLocaleString()} ms
                </p>
              </div>
            ) : (
              <p className="mt-3 text-sm text-slate-600">
                No prediction yet.
              </p>
            )}

            {error && (
              <p className="mt-3 text-sm text-red-600">{error}</p>
            )}
          </div>
        </div>
      </div>
    </main>
  );
}