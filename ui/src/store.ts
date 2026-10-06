import { reactive, ref } from 'vue'
import axios from 'axios'

export const store = reactive({
  backendStatus: 'offline',
  alarms: [] as any[],
  anomalies: [] as any[],
  watchlist: [] as any[],
  rcaRules: [] as any[],
  modelStatus: {} as Record<string, any>,
  modelVersions: {} as Record<string, any>,
  driftStatus: null as any,
  retrainState: {} as Record<string, any>,        // 트랙별 재학습 상태 (연속 드리프트 일수, 쿨다운)
  isCheckingDrift: false,
  isRefreshingAnomalies: false,
  isRefreshingWatchlist: false,
  isRefreshingModelStatus: false,
  _refreshTimer: null as any,

  debouncedFetch(delay = 1000) {
    if (this._refreshTimer) clearTimeout(this._refreshTimer)
    this._refreshTimer = setTimeout(() => {
      this.fetchAnomalies()
      this.fetchWatchlist()
      this._refreshTimer = null
      console.log('[Store] Dashboard data refreshed (globally debounced)')
    }, delay)
  },

  async fetchActiveAlarms() {
    try {
      // 최신 시점의 Critical 알람만 가져오기
      const res = await axios.get('/api/anomalies?severity_min=3')
      this.alarms = res.data.map((a: any) => ({
        type: 'ALARM',
        event_time: a.occur_date,
        ip_addr: a.ip_addr,
        slot_id: a.slot_id,
        port_id: a.port_id,
        message: a.anomaly_reason
      }))
      console.log('[Store] Active alarms synchronized from DB:', this.alarms.length)
    } catch (err) {
      console.error('Failed to sync active alarms', err)
    }
  },

  async fetchAnomalies() {
    this.isRefreshingAnomalies = true
    try {
      const res = await axios.get('/api/anomalies?severity_min=1')
      this.anomalies = res.data
      this.backendStatus = 'online'
    } catch (err) {
      console.error('Failed to fetch anomalies', err)
      this.backendStatus = 'offline'
    } finally {
      this.isRefreshingAnomalies = false
    }
  },

  async fetchWatchlist() {
    this.isRefreshingWatchlist = true
    try {
      const res = await axios.get('/api/anomalies?severity_max=2&rising_only=true')
      this.watchlist = res.data
    } catch (err) {
      console.error('Failed to fetch trend data', err)
    } finally {
      this.isRefreshingWatchlist = false
    }
  },
  async fetchHistory(params: { ip_addr: string, slot_id: number, port_id: number, days?: number }) {
    try {
      const res = await axios.get(`/api/anomalies/history`, {
        params: { ...params, days: params.days || 1 }
      })
      return res.data
    } catch (err) {
      console.error('Failed to fetch history in store', err)
      throw err
    }
  },

  async fetchModelStatus() {
    this.isRefreshingModelStatus = true
    try {
      const res = await axios.get('/api/model/status')
      this.modelStatus = res.data
    } catch (err) {
      console.error('Failed to fetch model status', err)
    } finally {
      this.isRefreshingModelStatus = false
    }
  },



  async fetchModelVersions() {
    try {
      const res = await axios.get('/api/model/versions')
      this.modelVersions = res.data
    } catch (err) {
      console.error('Failed to fetch model versions', err)
    }
  },

  async promoteModel(ft: string, version: string, force = false) {
    const res = await axios.post('/api/model/promote', null, { params: { ft, version, force } })
    await this.fetchModelVersions()
    return res.data
  },

  async rerunGate(ft: string, version: string) {
    const res = await axios.post('/api/model/gate', null, { params: { ft, version } })
    await this.fetchModelVersions()
    return res.data
  },

  async rollbackModel(ft: string) {
    const res = await axios.post('/api/model/rollback', null, { params: { ft } })
    await this.fetchModelVersions()
    return res.data
  },

  // date_params 가 비어 있으면 서버가 최근 구간을 포트 단위 홀드아웃으로 나눠 학습한다 (권장 기본).
  // exclude_suspect=false 이면 장애 의심 구간 자동 제외를 끈다.
  // alert_policy: 후보와 짝으로 저장할 알람 정책 프리셋(default / precision). 빈 값이면 서버가 활성 모델의 정책을 승계한다.
  async trainModel(ft: string, training_config: any = {}, date_params: any = {}, exclude_suspect = true, alert_policy = '') {
    try {
      const res = await axios.post(`/api/model/train`, training_config, {
        params: {
          ft,
          train_start: date_params.train_start,
          train_end: date_params.train_end,
          test_start: date_params.test_start,
          test_end: date_params.test_end,
          exclude_suspect,
          alert_policy: alert_policy || undefined
        }
      })
      return res.data
    } catch (err) {
      console.error('Failed to trigger training', err)
      throw err
    }
  },

  async stopTraining(ft: string) {
    try {
      const res = await axios.post(`/api/model/train/stop?ft=${ft}`)
      await this.fetchModelStatus()
      return res.data
    } catch (err) {
      console.error('Failed to stop training', err)
      throw err
    }
  },

  // --- RCA Rules Management ---
  async fetchRcaRules() {
    try {
      const res = await axios.get('/api/rca/rules')
      this.rcaRules = res.data.rules || []
    } catch (err) {
      console.error('Failed to fetch RCA rules', err)
    }
  },

  async addRcaRule(rule: any) {
    try {
      const res = await axios.post('/api/rca/rules', rule)
      await this.fetchRcaRules()
      return res.data
    } catch (err) {
      console.error('Failed to add RCA rule', err)
      throw err
    }
  },

  async deleteRcaRule(ruleId: string) {
    try {
      const res = await axios.delete(`/api/rca/rules/${ruleId}`)
      await this.fetchRcaRules()
      return res.data
    } catch (err) {
      console.error('Failed to delete RCA rule', err)
      throw err
    }
  },

  // --- Data Drift ---
  async fetchDriftStatus() {
    try {
      const res = await axios.get('/api/drift/status')
      this.driftStatus = res.data.last_result
      this.retrainState = res.data.retrain_state || {}
    } catch (err) {
      console.error('Failed to fetch drift status', err)
    }
  },

  async checkDrift() {
    this.isCheckingDrift = true
    try {
      const res = await axios.post('/api/drift/check')
      this.driftStatus = res.data
      await this.fetchDriftStatus()           // 지속 일수/쿨다운 갱신 (last_result 는 방금 결과와 같음)
      return res.data
    } catch (err) {
      console.error('Failed to check drift', err)
      throw err
    } finally {
      this.isCheckingDrift = false
    }
  }
})
