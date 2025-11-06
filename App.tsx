import { useState, useEffect } from 'react';
import { VitalsChart } from './components/VitalsChart';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from './components/ui/card';
import { Badge } from './components/ui/badge';
import { Alert, AlertDescription } from './components/ui/alert';
import { Activity, Heart, Droplet, Thermometer } from 'lucide-react';

interface VitalData {
  time: string;
  bpm: number;
  spo2: number;
  temperature: number;
  ecg: number;
}

export default function App() {
  const [vitalsData, setVitalsData] = useState<VitalData[]>([]);
  const [isConnected, setIsConnected] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [rawApiData, setRawApiData] = useState<any>(null);

  const fetchVitals = async () => {
    try {
      const response = await fetch('https://health-monitor-backend-6j1k.onrender.com/api/v1/vitals');
      
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      
      const data = await response.json();
      console.log('API Response:', data); // Debug: log the actual API response
      setRawApiData(data); // Store raw API data for debugging
      
      // Helper function to safely extract numeric value
      const extractNumber = (value: any): number => {
        if (value === null || value === undefined || value === '') return 0;
        const num = Number(value);
        return isNaN(num) ? 0 : num;
      };
      
      // Helper function to get BPM from various field names
      const getBpmValue = (item: any): number => {
        const bpmValue = extractNumber(item.bpm ?? item.heart_rate ?? item.heartRate ?? item.BPM ?? item.heartrate);
        console.log('Extracted BPM:', bpmValue, 'from:', { bpm: item.bpm, heart_rate: item.heart_rate, heartRate: item.heartRate });
        return bpmValue;
      };
      
      // Handle different API response structures
      let dataPoints: VitalData[] = [];
      
      if (Array.isArray(data)) {
        // If API returns an array of vitals, reverse it to process from last to first
        dataPoints = data.reverse().map((item: any) => ({
          time: item.time || new Date().toISOString(),
          bpm: getBpmValue(item),
          spo2: extractNumber(item.spo2 ?? item.SpO2 ?? item.spo2_percentage),
          temperature: extractNumber(item.temperature ?? item.temp),
          ecg: extractNumber(item.ecg ?? item.ECG),
        }));
      } else if (data.vitals && Array.isArray(data.vitals)) {
        // If API returns {vitals: [...]}, reverse it to process from last to first
        dataPoints = data.vitals.reverse().map((item: any) => ({
          time: item.time || new Date().toISOString(),
          bpm: getBpmValue(item),
          spo2: extractNumber(item.spo2 ?? item.SpO2 ?? item.spo2_percentage),
          temperature: extractNumber(item.temperature ?? item.temp),
          ecg: extractNumber(item.ecg ?? item.ECG),
        }));
      } else {
        // If API returns a single data point
        dataPoints = [{
          time: data.time || new Date().toISOString(),
          bpm: getBpmValue(data),
          spo2: extractNumber(data.spo2 ?? data.SpO2 ?? data.spo2_percentage),
          temperature: extractNumber(data.temperature ?? data.temp),
          ecg: extractNumber(data.ecg ?? data.ECG),
        }];
      }
      
      console.log('Processed data points (last to first):', dataPoints);
      
      setVitalsData(prev => {
        // If we received multiple data points, replace all data
        if (dataPoints.length > 1) {
          return dataPoints.slice(-50);
        }
        // Otherwise append the new data point
        const updated = [...prev, ...dataPoints];
        return updated.slice(-50);
      });
      
      setIsConnected(true);
      setError(null);
      setIsLoading(false);
    } catch (err) {
      console.error('Error fetching vitals:', err);
      setError('Failed to connect to smartwatch. Retrying...');
      setIsConnected(false);
      setIsLoading(false);
    }
  };

  useEffect(() => {
    // Initial fetch
    fetchVitals();
    
    // Poll for new data every 2 seconds
    const interval = setInterval(fetchVitals, 2000);
    
    return () => clearInterval(interval);
  }, []);

  const currentVitals = vitalsData[vitalsData.length - 1];

  const getHeartRateStatus = (bpm: number) => {
    if (bpm < 60) return { status: 'Low', variant: 'secondary' as const };
    if (bpm > 100) return { status: 'High', variant: 'destructive' as const };
    return { status: 'Normal', variant: 'default' as const };
  };

  const getSpo2Status = (spo2: number) => {
    if (spo2 < 95) return { status: 'Low', variant: 'destructive' as const };
    return { status: 'Normal', variant: 'default' as const };
  };

  const getTemperatureStatus = (temp: number) => {
    if (temp < 36.1) return { status: 'Low', variant: 'secondary' as const };
    if (temp > 37.2) return { status: 'High', variant: 'destructive' as const };
    return { status: 'Normal', variant: 'default' as const };
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 to-slate-100 p-4 md:p-8">
      <div className="max-w-7xl mx-auto space-y-6">
        {/* Header */}
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-slate-900">Health Monitor Dashboard</h1>
            <p className="text-slate-600">Real-time vital signs from smartwatch</p>
          </div>
          <div className="flex items-center gap-2">
            <div className={`h-3 w-3 rounded-full ${isConnected ? 'bg-green-500' : 'bg-red-500'} animate-pulse`} />
            <span className="text-slate-700">{isConnected ? 'Connected' : 'Disconnected'}</span>
          </div>
        </div>

        {/* Error Alert */}
        {error && (
          <Alert variant="destructive">
            <AlertDescription>{error}</AlertDescription>
          </Alert>
        )}

        {/* Current Vitals Cards */}
        {currentVitals && !isLoading && (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-5 gap-4">
            <Card>
              <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                <CardTitle className="text-slate-900">Heart Rate</CardTitle>
                <Heart className="h-5 w-5 text-red-500" />
              </CardHeader>
              <CardContent>
                <div className="flex items-baseline gap-2">
                  <span className="text-slate-900">{currentVitals.bpm}</span>
                  <span className="text-slate-600">BPM</span>
                </div>
                <Badge variant={getHeartRateStatus(currentVitals.bpm).variant} className="mt-2">
                  {getHeartRateStatus(currentVitals.bpm).status}
                </Badge>
              </CardContent>
            </Card>

            <Card>
              <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                <CardTitle className="text-slate-900">Blood Oxygen</CardTitle>
                <Droplet className="h-5 w-5 text-blue-500" />
              </CardHeader>
              <CardContent>
                <div className="flex items-baseline gap-2">
                  <span className="text-slate-900">{currentVitals.spo2}</span>
                  <span className="text-slate-600">%</span>
                </div>
                <Badge variant={getSpo2Status(currentVitals.spo2).variant} className="mt-2">
                  {getSpo2Status(currentVitals.spo2).status}
                </Badge>
              </CardContent>
            </Card>

            <Card>
              <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                <CardTitle className="text-slate-900">Temperature</CardTitle>
                <Thermometer className="h-5 w-5 text-orange-500" />
              </CardHeader>
              <CardContent>
                <div className="flex items-baseline gap-2">
                  <span className="text-slate-900">{currentVitals.temperature.toFixed(1)}</span>
                  <span className="text-slate-600">°C</span>
                </div>
                <Badge variant={getTemperatureStatus(currentVitals.temperature).variant} className="mt-2">
                  {getTemperatureStatus(currentVitals.temperature).status}
                </Badge>
              </CardContent>
            </Card>

            <Card>
              <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                <CardTitle className="text-slate-900">ECG</CardTitle>
                <Activity className="h-5 w-5 text-purple-500" />
              </CardHeader>
              <CardContent>
                <div className="flex items-baseline gap-2">
                  <span className="text-slate-900">{currentVitals.ecg.toFixed(3)}</span>
                  <span className="text-slate-600">mV</span>
                </div>
                <div className="flex items-center gap-1 mt-2 text-slate-600">
                  <Activity className="h-4 w-4" />
                  <span>Monitoring</span>
                </div>
              </CardContent>
            </Card>
          </div>
        )}

        {/* Charts */}
        <div className="space-y-6">
          <VitalsChart
            data={vitalsData}
            dataKey="bpm"
            title="Heart Rate (BPM)"
            description="Beats per minute over time"
            color="#ef4444"
            icon={<Heart className="h-5 w-5 text-red-500" />}
            unit="BPM"
          />

          <VitalsChart
            data={vitalsData}
            dataKey="spo2"
            title="Blood Oxygen Level (SpO2%)"
            description="Oxygen saturation percentage over time"
            color="#3b82f6"
            icon={<Droplet className="h-5 w-5 text-blue-500" />}
            unit="%"
          />

          <VitalsChart
            data={vitalsData}
            dataKey="temperature"
            title="Body Temperature"
            description="Temperature readings over time"
            color="#f97316"
            icon={<Thermometer className="h-5 w-5 text-orange-500" />}
            unit="°C"
          />

          <VitalsChart
            data={vitalsData}
            dataKey="ecg"
            title="ECG Voltage Output"
            description="Electrocardiogram voltage readings over time"
            color="#a855f7"
            icon={<Activity className="h-5 w-5 text-purple-500" />}
            unit="mV"
          />
        </div>

        {/* Footer Info */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <Card className="bg-slate-50 border-slate-200">
            <CardHeader>
              <CardTitle className="text-slate-900">Sensor Information</CardTitle>
              <CardDescription>Data is automatically refreshed every 2 seconds from the smartwatch sensors</CardDescription>
            </CardHeader>
            <CardContent className="text-slate-600">
              <p>Last updated: {currentVitals ? currentVitals.time : 'Waiting for data...'}</p>
              <p className="mt-2">Total data points: {vitalsData.length}</p>
            </CardContent>
          </Card>

          <Card className="bg-slate-50 border-slate-200">
            <CardHeader>
              <CardTitle className="text-slate-900">Raw API Response</CardTitle>
              <CardDescription>Latest data structure from the API (for debugging)</CardDescription>
            </CardHeader>
            <CardContent>
              <pre className="text-slate-600 overflow-auto max-h-40 p-2 bg-white rounded border border-slate-200">
                {rawApiData ? JSON.stringify(rawApiData, null, 2) : 'No data yet...'}
              </pre>
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  );
}
