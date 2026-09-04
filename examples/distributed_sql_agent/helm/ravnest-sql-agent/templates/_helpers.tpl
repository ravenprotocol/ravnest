{{/*
Expand the name of the chart.
*/}}
{{- define "ravnest-sql-agent.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Create a default fully qualified app name.
*/}}
{{- define "ravnest-sql-agent.fullname" -}}
{{- if .Values.fullnameOverride }}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- $name := default .Chart.Name .Values.nameOverride }}
{{- printf "%s-%s" .Release.Name $name | trunc 63 | trimSuffix "-" }}
{{- end }}
{{- end }}

{{/*
Shared labels applied to every resource.
*/}}
{{- define "ravnest-sql-agent.labels" -}}
helm.sh/chart: {{ printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" }}
{{ include "ravnest-sql-agent.selectorLabels" . }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end }}

{{/*
Selector labels used in matchLabels / selector blocks.
*/}}
{{- define "ravnest-sql-agent.selectorLabels" -}}
app.kubernetes.io/name: {{ include "ravnest-sql-agent.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}

{{/*
Compute the agent model — inherit from backend if not explicitly set.
*/}}
{{- define "ravnest-sql-agent.agentModel" -}}
{{- if .Values.agent.model }}
{{- .Values.agent.model }}
{{- else if eq .Values.compute.backend "vllm" }}
{{- .Values.compute.vllm.model }}
{{- else if eq .Values.compute.backend "ravnest" }}
{{- .Values.compute.ravnest.model }}
{{- else }}
{{- .Values.compute.ollama.model }}
{{- end }}
{{- end }}

{{/*
Compute service URL for inter-pod mesh communication.
*/}}
{{- define "ravnest-sql-agent.computeServiceUrl" -}}
{{- printf "http://%s-compute:%d" (include "ravnest-sql-agent.fullname" .) (.Values.compute.port | int) }}
{{- end }}

{{/*
SQL service URL for inter-pod mesh communication.
*/}}
{{- define "ravnest-sql-agent.sqlServiceUrl" -}}
{{- printf "http://%s-sql:%d" (include "ravnest-sql-agent.fullname" .) (.Values.sql.port | int) }}
{{- end }}

{{/*
In-cluster Ollama URL (only meaningful when backend=ollama and serviceUrl is empty).
*/}}
{{- define "ravnest-sql-agent.ollamaUrl" -}}
{{- if .Values.compute.ollama.serviceUrl }}
{{- .Values.compute.ollama.serviceUrl }}
{{- else }}
{{- printf "http://%s-ollama:%d" (include "ravnest-sql-agent.fullname" .) (.Values.ollama.port | int) }}
{{- end }}
{{- end }}

{{/*
Ravnest master address: DNS of StatefulSet pod-0 via headless service.
*/}}
{{- define "ravnest-sql-agent.ravnestMasterAddr" -}}
{{- printf "%s-compute-0.%s-compute-headless.%s.svc.cluster.local" (include "ravnest-sql-agent.fullname" .) (include "ravnest-sql-agent.fullname" .) .Release.Namespace }}
{{- end }}
