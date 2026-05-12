<template>
  <div class="space-y-6">
    <!-- Status Cards -->
    <div class="grid grid-cols-1 md:grid-cols-3 gap-4 mb-2">
      <div class="bg-slate-800 p-4 rounded-xl border border-slate-700 flex justify-between items-center relative overflow-hidden">
        <div class="z-10">
          <p class="text-slate-400 text-xs uppercase font-bold tracking-wider mb-1">Anomaly Detection Scheduler</p>
          <div class="flex items-center gap-3">
            <span :class="['text-xl font-mono font-bold uppercase', 
                          store.schedulerStatus === 'running' ? 'text-emerald-400' : 'text-rose-400']">
              {{ store.schedulerStatus }}
            </span>
            <div class="flex gap-1">
              <button @click="store.controlScheduler('start')" 
                      v-if="store.schedulerStatus !== 'running'"
                      class="p-1.5 bg-emerald-500/10 hover:bg-emerald-500/20 text-emerald-400 rounded-lg transition-colors border border-emerald-500/20" title="Start Engine">
                <svg xmlns="http://www.w3.org/2000/svg" class="w-4 h-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><polygon points="5 3 19 12 5 21 5 3"></polygon></svg>
              </button>
              <button @click="store.controlScheduler('stop')" 
                      v-if="store.schedulerStatus === 'running'"
                      class="p-1.5 bg-rose-500/10 hover:bg-rose-500/20 text-rose-400 rounded-lg transition-colors border border-rose-500/20" title="Stop Engine">
                <svg xmlns="http://www.w3.org/2000/svg" class="w-4 h-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><rect x="6" y="4" width="4" height="16"></rect><rect x="14" y="4" width="4" height="16"></rect></svg>
              </button>
            </div>
          </div>
        </div>
        <div class="flex items-center gap-3 z-10">
          <div v-if="store.schedulerStatus === 'running'" class="text-right">
            <p class="text-[10px] text-slate-500 uppercase font-bold tracking-tighter">Next Run</p>
            <p class="text-sm text-blue-400 font-mono font-bold leading-none">{{ store.nextRunTime || '--:--' }}</p>
          </div>
          <div class="p-3 bg-slate-700/30 rounded-lg">
            <svg xmlns="http://www.w3.org/2000/svg" :class="['w-6 h-6', store.schedulerStatus === 'running' ? 'text-emerald-400' : 'text-slate-500']" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"></circle><polyline points="12 6 12 12 16 14"></polyline></svg>
          </div>
        </div>
      </div>
      <div class="bg-slate-800 p-4 rounded-xl border border-slate-700 flex justify-between items-center">
        <div>
          <p class="text-slate-400 text-xs uppercase font-bold tracking-wider mb-1">Total Anomalies</p>
          <p class="text-xl font-mono text-emerald-400">{{ store.anomalies.length }}</p>
        </div>
        <div class="p-3 bg-emerald-500/10 rounded-lg">
          <svg xmlns="http://www.w3.org/2000/svg" class="w-6 h-6 text-emerald-400" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M22 12h-4l-3 9L9 3l-3 9H2"></path></svg>
        </div>
      </div>
      <div class="bg-slate-800 p-4 rounded-xl border border-slate-700 flex justify-between items-center">
        <div>
          <p class="text-slate-400 text-xs uppercase font-bold tracking-wider mb-1">Critical Alarms (History)</p>
          <p class="text-xl font-mono text-rose-400">{{ store.alarms.length }}</p>
        </div>
        <div class="p-3 bg-rose-500/10 rounded-lg">
          <svg xmlns="http://www.w3.org/2000/svg" class="w-6 h-6 text-rose-400" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"></path><line x1="12" y1="9" x2="12" y2="13"></line><line x1="12" y1="17" x2="12.01" y2="17"></line></svg>
        </div>
      </div>
    </div>

    <div class="grid grid-cols-1 lg:grid-cols-2 gap-6">
      <!-- Recent Anomalies -->
      <div class="bg-slate-800 p-6 rounded-xl border border-slate-700 shadow-xl h-[580px] flex flex-col">
        <div class="flex justify-between items-center mb-4">
          <div class="flex items-center gap-3">
            <h2 class="text-lg font-semibold flex items-center gap-2">
              <svg xmlns="http://www.w3.org/2000/svg" class="w-5 h-5 text-rose-400" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"></path><line x1="12" y1="9" x2="12" y2="13"></line><line x1="12" y1="17" x2="12.01" y2="17"></line></svg>
              Recent Anomalies
            </h2>
            <button @click="store.fetchAnomalies" 
                    :disabled="store.isRefreshingAnomalies"
                    class="p-1.5 text-slate-400 hover:text-blue-400 hover:bg-blue-500/10 rounded-lg transition-all"
                    title="Refresh Anomalies">
              <svg xmlns="http://www.w3.org/2000/svg" :class="['w-4 h-4', store.isRefreshingAnomalies ? 'animate-spin text-blue-400' : '']" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="23 4 23 10 17 10"></polyline><path d="M20.49 15a9 9 0 1 1-2.12-9.36L23 10"></path></svg>
            </button>
          </div>
          <!-- Pagination Controls -->
          <div class="flex items-center gap-2" v-if="store.anomalies.length > 0">
            <button @click="anomaliesPage--" :disabled="anomaliesPage === 1" 
                    class="p-1.5 bg-slate-700 rounded hover:bg-slate-600 disabled:opacity-30 transition-colors">
              <svg xmlns="http://www.w3.org/2000/svg" class="w-4 h-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="15 18 9 12 15 6"></polyline></svg>
            </button>
            <span class="text-xs font-mono text-slate-400">{{ anomaliesPage }} / {{ Math.ceil(store.anomalies.length / pageSize) }}</span>
            <button @click="anomaliesPage++" :disabled="anomaliesPage >= Math.ceil(store.anomalies.length / pageSize)" 
                    class="p-1.5 bg-slate-700 rounded hover:bg-slate-600 disabled:opacity-30 transition-colors">
              <svg xmlns="http://www.w3.org/2000/svg" class="w-4 h-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="9 18 15 12 9 6"></polyline></svg>
            </button>
          </div>
        </div>
        <div class="overflow-x-auto flex-1">
          <table class="w-full text-left text-sm">
            <thead class="bg-slate-700/30 text-slate-400 uppercase text-[11px] tracking-wider">
              <tr>
                <th class="px-4 py-3">Time</th>
                <th class="px-4 py-3">IP Address</th>
                <th class="px-4 py-3">Port</th>
                <th class="px-4 py-3">Severity</th>
                <th class="px-4 py-3">Reason</th>
                <th class="px-4 py-3 text-right">Actions</th>
              </tr>
            </thead>
            <tbody class="divide-y divide-slate-700">
              <tr v-for="item in pagedAnomalies" :key="item.occur_date + item.ip_addr + item.slot_id + item.port_id" class="hover:bg-slate-700/20 transition-colors">
                <td class="px-4 py-3 font-mono text-xs">{{ item.occur_date }}</td>
                <td class="px-4 py-3 font-semibold">{{ item.ip_addr }}</td>
                <td class="px-4 py-3 text-slate-400">S{{ item.slot_id }}/P{{ item.port_id }}</td>
                <td class="px-4 py-3">
                  <span :class="['px-2 py-0.5 rounded text-[10px] font-bold', 
                                item.alarm_label === 'CRITICAL' ? 'bg-rose-500/20 text-rose-400 border border-rose-500/30' : 
                                item.alarm_label === 'MAJOR' ? 'bg-orange-500/20 text-orange-400 border border-orange-500/30' : 
                                'bg-blue-500/20 text-blue-400 border border-blue-500/30']">
                    {{ item.alarm_label }}
                  </span>
                </td>
                <td class="px-4 py-3 text-xs text-slate-400 truncate max-w-[200px]" :title="item.anomaly_reason">
                  {{ item.anomaly_reason }}
                </td>
                <td class="px-4 py-3 text-right">
                  <button @click="openGraph(item)" class="text-blue-400 hover:text-blue-300 text-xs font-bold flex items-center gap-1 ml-auto">
                    <svg xmlns="http://www.w3.org/2000/svg" class="w-3.5 h-3.5" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><line x1="18" y1="20" x2="18" y2="10"></line><line x1="12" y1="20" x2="12" y2="4"></line><line x1="6" y1="20" x2="6" y2="14"></line></svg>
                    Graph
                  </button>
                </td>
              </tr>
              <tr v-if="pagedAnomalies.length === 0">
                <td colspan="6" class="px-4 py-10 text-center text-slate-500 italic">No anomalies detected. Everything looks stable.</td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>

      <!-- Watchlist (Proactive Warnings) -->
      <div class="bg-slate-800 p-6 rounded-xl border border-slate-700 shadow-xl border-l-4 border-l-amber-500">
        <div class="flex justify-between items-center mb-4">
          <div class="flex items-center gap-3">
            <h2 class="text-lg font-semibold flex items-center gap-2">
              <svg xmlns="http://www.w3.org/2000/svg" class="w-5 h-5 text-amber-500" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M18 8A6 6 0 0 0 6 8c0 7-3 9-3 9h18s-3-2-3-9"></path><path d="M13.73 21a2 2 0 0 1-3.46 0"></path></svg>
              Watchlist (Rising Trends)
            </h2>
            <button @click="store.fetchWatchlist" 
                    :disabled="store.isRefreshingWatchlist"
                    class="p-1.5 text-slate-400 hover:text-amber-400 hover:bg-amber-500/10 rounded-lg transition-all"
                    title="Refresh Watchlist">
              <svg xmlns="http://www.w3.org/2000/svg" :class="['w-4 h-4', store.isRefreshingWatchlist ? 'animate-spin text-amber-400' : '']" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="23 4 23 10 17 10"></polyline><path d="M20.49 15a9 9 0 1 1-2.12-9.36L23 10"></path></svg>
            </button>
          </div>
          <!-- Pagination Controls -->
          <div class="flex items-center gap-2" v-if="store.watchlist.length > 0">
            <button @click="watchlistPage--" :disabled="watchlistPage === 1" 
                    class="p-1.5 bg-slate-700 rounded hover:bg-slate-600 disabled:opacity-30 transition-colors">
              <svg xmlns="http://www.w3.org/2000/svg" class="w-4 h-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="15 18 9 12 15 6"></polyline></svg>
            </button>
            <span class="text-xs font-mono text-slate-400">{{ watchlistPage }} / {{ Math.ceil(store.watchlist.length / pageSize) }}</span>
            <button @click="watchlistPage++" :disabled="watchlistPage >= Math.ceil(store.watchlist.length / pageSize)" 
                    class="p-1.5 bg-slate-700 rounded hover:bg-slate-600 disabled:opacity-30 transition-colors">
              <svg xmlns="http://www.w3.org/2000/svg" class="w-4 h-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="9 18 15 12 9 6"></polyline></svg>
            </button>
          </div>
        </div>
        <div class="overflow-x-auto min-h-[480px]">
          <table class="w-full text-left text-sm">
            <thead class="bg-slate-700/30 text-slate-400 uppercase text-[11px] tracking-wider">
              <tr>
                <th class="px-4 py-3">Time</th>
                <th class="px-4 py-3">IP Address</th>
                <th class="px-4 py-3">Port</th>
                <th class="px-4 py-3">Severity</th>
                <th class="px-4 py-3">Trend</th>
                <th class="px-4 py-3">Estimated Time to Critical</th>
              </tr>
            </thead>
            <tbody class="divide-y divide-slate-700">
              <tr v-for="item in pagedWatchlist" :key="'watch-' + item.ip_addr + item.slot_id + item.port_id + item.occur_date" class="hover:bg-slate-700/20 transition-colors">
                <td class="px-4 py-3 font-mono text-xs text-slate-400">{{ item.occur_date }}</td>
                <td class="px-4 py-3 font-semibold">{{ item.ip_addr }}</td>
                <td class="px-4 py-3 text-slate-400">S{{ item.slot_id }}/P{{ item.port_id }}</td>
                <td class="px-4 py-3">
                  <div class="flex flex-col gap-1.5 min-w-[120px]">
                    <div class="flex justify-between items-center">
                      <span :class="['px-1.5 py-0.5 rounded text-[9px] font-bold uppercase', 
                                    item.alarm_label === 'MAJOR' ? 'bg-orange-500/20 text-orange-400 border border-orange-500/30' : 
                                    item.alarm_label === 'MINOR' ? 'bg-blue-500/20 text-blue-400 border border-blue-500/30' : 
                                    'bg-slate-500/20 text-slate-400 border border-slate-500/30']">
                        {{ item.alarm_label }}
                      </span>
                      <span class="text-[10px] font-mono text-slate-400">{{ item.severity.toFixed(1) }}%</span>
                    </div>
                    <div class="w-full bg-slate-700 rounded-full h-1">
                      <div :class="['h-1 rounded-full transition-all duration-500', 
                                   item.severity > 70 ? 'bg-orange-500' : 
                                   item.severity > 40 ? 'bg-blue-500' : 'bg-emerald-500']" 
                           :style="{ width: item.severity + '%' }"></div>
                    </div>
                  </div>
                </td>
                <td class="px-4 py-3">
                  <span class="flex items-center gap-1 text-amber-400 font-bold text-xs">
                    <svg xmlns="http://www.w3.org/2000/svg" class="w-3 h-3" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"><polyline points="23 6 13.5 15.5 8.5 10.5 1 18"></polyline><polyline points="17 6 23 6 23 12"></polyline></svg>
                    +{{ item.slope.toFixed(2) }}
                  </span>
                </td>
                <td class="px-4 py-3 font-mono text-xs">
                  <div v-if="item.ttf_minutes" class="flex flex-col gap-0.5">
                    <span class="text-blue-400 font-bold">{{ item.expected_fatal_time }}</span>
                    <span class="text-[10px] text-slate-500">({{ formatTTF(item.ttf_minutes) }} later)</span>
                  </div>
                  <span v-else class="text-slate-500 italic">Calculating...</span>
                </td>
              </tr>
              <tr v-if="pagedWatchlist.length === 0">
                <td colspan="6" class="px-4 py-10 text-center text-slate-500 italic">No rising trends detected.</td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>
    </div>

    <!-- Graph Modal -->
    <div v-if="selectedPort" class="fixed inset-0 bg-slate-950/80 backdrop-blur-sm z-50 flex items-center justify-center p-4">
      <div class="bg-slate-800 border border-slate-700 rounded-2xl w-full max-w-5xl max-h-[90vh] overflow-hidden flex flex-col shadow-2xl animate-in fade-in zoom-in duration-200">
        <div class="p-6 border-b border-slate-700 flex justify-between items-center">
          <div>
            <h3 class="text-xl font-bold text-blue-400">{{ selectedPort.ip_addr }} (S{{ selectedPort.slot_id }}/P{{ selectedPort.port_id }})</h3>
            <p class="text-sm text-slate-400">Time-series Analysis & Anomaly Scores (Last 24h)</p>
          </div>
          <button @click="closeGraph" class="p-2 hover:bg-slate-700 rounded-full transition-colors">
            <svg xmlns="http://www.w3.org/2000/svg" class="w-6 h-6" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="18" y1="6" x2="6" y2="18"></line><line x1="6" y1="6" x2="18" y2="18"></line></svg>
          </button>
        </div>
        
        <div class="p-6 flex-1 overflow-y-auto space-y-8 bg-slate-900/50">
          <div v-if="isLoadingHistory" class="flex flex-col items-center justify-center py-20 gap-4">
            <div class="w-10 h-10 border-4 border-blue-500/20 border-t-blue-500 rounded-full animate-spin"></div>
            <p class="text-slate-400 animate-pulse">Loading port history...</p>
          </div>
          
          <template v-else-if="historyData.length > 0">
            <!-- Anomaly Score Chart -->
            <div class="bg-slate-800 p-4 rounded-xl border border-slate-700">
              <h4 class="text-xs font-bold uppercase tracking-widest text-slate-500 mb-4">Anomaly Score & Threshold</h4>
              <div class="h-[250px]">
                <Line :data="scoreChartData" :options="chartOptions" />
              </div>
            </div>

            <!-- Traffic Metrics Chart -->
            <div v-if="hasTrafficData" class="bg-slate-800 p-4 rounded-xl border border-slate-700">
              <h4 class="text-xs font-bold uppercase tracking-widest text-slate-500 mb-4">Traffic Performance (Packets)</h4>
              <div class="h-[250px]">
                <Line :data="trafficChartData" :options="chartOptions" />
              </div>
            </div>

            <!-- Optical Metrics Chart -->
            <div v-if="hasOpticalData" class="bg-slate-800 p-4 rounded-xl border border-slate-700">
              <h4 class="text-xs font-bold uppercase tracking-widest text-slate-500 mb-4">Optical Power (dBm)</h4>
              <div class="h-[250px]">
                <Line :data="opticalChartData" :options="chartOptions" />
              </div>
            </div>
          </template>

          <div v-else class="text-center py-20 text-slate-500 italic border-2 border-dashed border-slate-700 rounded-2xl">
            No history data accumulated yet for this port.
          </div>
        </div>
        
        <div class="p-4 bg-slate-800 border-t border-slate-700 flex justify-end">
          <button @click="closeGraph" class="px-6 py-2 bg-slate-700 hover:bg-slate-600 rounded-lg font-bold transition-colors">Close</button>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, computed } from 'vue'
import axios from 'axios'
import { store } from '../store'
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  Filler
} from 'chart.js'
import { Line } from 'vue-chartjs'

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  Filler
)

// Graph State
const selectedPort = ref<any>(null)
const historyData = ref<any[]>([])
const isLoadingHistory = ref(false)

// Pagination State
const pageSize = 10
const anomaliesPage = ref(1)
const watchlistPage = ref(1)

const pagedAnomalies = computed(() => {
  const start = (anomaliesPage.value - 1) * pageSize
  return store.anomalies.slice(start, start + pageSize)
})

const pagedWatchlist = computed(() => {
  const start = (watchlistPage.value - 1) * pageSize
  return store.watchlist.slice(start, start + pageSize)
})

const formatTTF = (minutes: number | null) => {
  if (minutes === null || minutes === undefined) return 'N/A'
  const h = Math.floor(minutes / 60)
  const m = Math.round(minutes % 60)
  return h > 0 ? `${h}h ${m}m` : `${m}m`
}

const scoreChartData = computed(() => ({
  labels: historyData.value.map(d => d.occur_date.split(' ')[1].substring(0, 5)),
  datasets: [
    {
      label: 'Severity (%)',
      data: historyData.value.map(d => d.severity),
      borderColor: '#f43f5e',
      backgroundColor: 'rgba(244, 63, 94, 0.1)',
      fill: true,
      tension: 0.4,
      pointRadius: 2,
    },
    {
      label: 'Anomaly Score',
      data: historyData.value.map(d => d.anomaly_score * 10),
      borderColor: '#3b82f6',
      borderDash: [5, 5],
      tension: 0.4,
      pointRadius: 0,
    },
    {
      label: 'Threshold',
      data: historyData.value.map(d => d.threshold * 10),
      borderColor: '#94a3b8',
      borderDash: [2, 2],
      pointRadius: 0,
      fill: false,
    }
  ]
}))

const hasTrafficData = computed(() => historyData.value.some(d => d.tx_packet > 0 || d.rx_packet > 0 || d.error_packet > 0))
const hasOpticalData = computed(() => historyData.value.some(d => d.tx_avg_power !== 0 || d.rx_avg_power !== 0))

const trafficChartData = computed(() => ({
  labels: historyData.value.map(d => d.occur_date.split(' ')[1].substring(0, 5)),
  datasets: [
    {
      label: 'TX Packets',
      data: historyData.value.map(d => d.tx_packet),
      borderColor: '#10b981',
      tension: 0.4,
    },
    {
      label: 'RX Packets',
      data: historyData.value.map(d => d.rx_packet),
      borderColor: '#3b82f6',
      tension: 0.4,
    },
    {
      label: 'Errors',
      data: historyData.value.map(d => d.error_packet),
      borderColor: '#f43f5e',
      backgroundColor: 'rgba(244, 63, 94, 0.5)',
      fill: true,
      tension: 0.4,
    }
  ]
}))

const opticalChartData = computed(() => ({
  labels: historyData.value.map(d => d.occur_date.split(' ')[1].substring(0, 5)),
  datasets: [
    {
      label: 'RX Power (dBm)',
      data: historyData.value.map(d => d.rx_avg_power),
      borderColor: '#a855f7',
      tension: 0.4,
    },
    {
      label: 'TX Power (dBm)',
      data: historyData.value.map(d => d.tx_avg_power),
      borderColor: '#eab308',
      tension: 0.4,
    }
  ]
}))

const chartOptions = {
  responsive: true,
  maintainAspectRatio: false,
  plugins: {
    legend: {
      position: 'top' as const,
      labels: { color: '#94a3b8', boxWidth: 12, usePointStyle: true, font: { size: 10 } }
    },
    tooltip: {
      mode: 'index' as const,
      intersect: false,
      backgroundColor: '#1e293b',
      titleColor: '#f8fafc',
      bodyColor: '#cbd5e1',
      borderColor: '#334155',
      borderWidth: 1,
    }
  },
  scales: {
    y: {
      grid: { color: 'rgba(148, 163, 184, 0.05)' },
      ticks: { color: '#64748b', font: { size: 10 } }
    },
    x: {
      grid: { display: false },
      ticks: { color: '#64748b', font: { size: 10 } }
    }
  }
}

const openGraph = async (port: any) => {
  selectedPort.value = port
  isLoadingHistory.value = true
  try {
    historyData.value = await store.fetchHistory({
      ip_addr: port.ip_addr,
      slot_id: port.slot_id,
      port_id: port.port_id,
      days: 1
    })
  } catch (err) {
    console.error('Failed to load history', err)
  } finally {
    isLoadingHistory.value = false
  }
}

const closeGraph = () => {
  selectedPort.value = null
  historyData.value = []
}
</script>
