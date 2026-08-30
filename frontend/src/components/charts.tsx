import { useEffect, useState } from 'react';
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Line,
  LineChart,
  Pie,
  PieChart,
  ReferenceLine,
  ResponsiveContainer,
  Scatter,
  ScatterChart,
  Tooltip as RTooltip,
  XAxis,
  YAxis,
} from 'recharts';
import { useTheme } from '../theme/ThemeProvider';

/**
 * Recharts needs concrete colour values, not CSS variables, so the palette is read from
 * the document once per theme change. Charts therefore stay in step with the toggle
 * without any component hard-coding a hex value.
 */
export function usePalette() {
  const { theme } = useTheme();
  const [palette, setPalette] = useState<Record<string, string>>({});

  useEffect(() => {
    const styles = getComputedStyle(document.documentElement);
    const read = (name: string) => styles.getPropertyValue(name).trim();
    setPalette({
      series: [1, 2, 3, 4, 5, 6].map((i) => read(`--chart-${i}`)).join(','),
      grid: read('--chart-grid'),
      text: read('--text-muted'),
      surface: read('--surface'),
      border: read('--border'),
      primary: read('--chart-1'),
      action: read('--chart-2'),
      accent: read('--chart-3'),
      success: read('--chart-4'),
      danger: read('--danger'),
    });
  }, [theme]);

  return palette;
}

export function seriesColors(palette: Record<string, string>): string[] {
  return (palette.series ?? '').split(',').filter(Boolean);
}

function ChartTooltip({ palette }: { palette: Record<string, string> }) {
  return (
    <RTooltip
      cursor={{ fill: palette.grid, fillOpacity: 0.25 }}
      contentStyle={{
        background: palette.surface,
        border: `1px solid ${palette.border}`,
        borderRadius: 6,
        fontSize: 12,
        color: palette.text,
      }}
      labelStyle={{ color: palette.text, fontWeight: 600 }}
    />
  );
}

const AXIS_PROPS = { fontSize: 11, tickLine: false, axisLine: false } as const;

/** Shorten a category label for an axis without losing which one it is. */
function truncateLabel(value: unknown, max = 28): string {
  const text = String(value ?? '');
  return text.length > max ? `${text.slice(0, max - 1)}…` : text;
}

/** Horizontal bar chart — the default for ranked feature lists. */
export function RankedBarChart({
  data,
  xKey,
  yKey,
  height = 300,
  color,
  diverging,
  labelWidth = 150,
}: {
  data: Record<string, any>[];
  xKey: string;
  yKey: string;
  height?: number;
  color?: string;
  /** Colour negative values differently, for signed contributions. */
  diverging?: boolean;
  /** Space reserved for category labels. Long free-text values are truncated to fit. */
  labelWidth?: number;
}) {
  const palette = usePalette();
  const base = color ?? palette.primary;

  // Axis labels are truncated, never wrapped: a free-text column produces labels
  // hundreds of characters long, and letting them render in full turns the axis into
  // unreadable overlapping text. The tooltip still shows the full value.
  const maxChars = Math.max(8, Math.floor(labelWidth / 6.2));

  return (
    <ResponsiveContainer width="100%" height={height}>
      <BarChart data={data} layout="vertical" margin={{ left: 8, right: 16, top: 4, bottom: 4 }}>
        <CartesianGrid horizontal={false} stroke={palette.grid} />
        <XAxis type="number" stroke={palette.text} {...AXIS_PROPS} />
        <YAxis
          type="category"
          dataKey={yKey}
          stroke={palette.text}
          width={labelWidth}
          tickFormatter={(value) => truncateLabel(value, maxChars)}
          {...AXIS_PROPS}
        />
        <ChartTooltip palette={palette} />
        {diverging && <ReferenceLine x={0} stroke={palette.border} />}
        <Bar dataKey={xKey} radius={[0, 3, 3, 0]} maxBarSize={18}>
          {data.map((row, index) => (
            <Cell
              key={index}
              fill={diverging && row[xKey] < 0 ? palette.action : base}
            />
          ))}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  );
}

/** Vertical bar chart for category frequency distributions. */
export function CategoryBarChart({
  data,
  xKey,
  yKey,
  height = 240,
  color,
}: {
  data: Record<string, any>[];
  xKey: string;
  yKey: string;
  height?: number;
  color?: string;
}) {
  const palette = usePalette();
  return (
    <ResponsiveContainer width="100%" height={height}>
      <BarChart data={data} margin={{ left: 0, right: 8, top: 4, bottom: 4 }}>
        <CartesianGrid vertical={false} stroke={palette.grid} />
        <XAxis
          dataKey={xKey}
          stroke={palette.text}
          {...AXIS_PROPS}
          interval={0}
          angle={-30}
          textAnchor="end"
          height={72}
          tickFormatter={(value) => truncateLabel(value, 18)}
        />
        <YAxis stroke={palette.text} {...AXIS_PROPS} />
        <ChartTooltip palette={palette} />
        <Bar dataKey={yKey} fill={color ?? palette.primary} radius={[3, 3, 0, 0]} maxBarSize={40} />
      </BarChart>
    </ResponsiveContainer>
  );
}

/** Donut chart for class balance. */
export function DonutChart({
  data,
  nameKey,
  valueKey,
  height = 240,
}: {
  data: Record<string, any>[];
  nameKey: string;
  valueKey: string;
  height?: number;
}) {
  const palette = usePalette();
  const colors = seriesColors(palette);

  return (
    <ResponsiveContainer width="100%" height={height}>
      <PieChart>
        <Pie
          data={data}
          dataKey={valueKey}
          nameKey={nameKey}
          innerRadius="55%"
          outerRadius="80%"
          paddingAngle={2}
          stroke={palette.surface}
        >
          {data.map((_, index) => (
            <Cell key={index} fill={colors[index % colors.length]} />
          ))}
        </Pie>
        <ChartTooltip palette={palette} />
      </PieChart>
    </ResponsiveContainer>
  );
}

/** Line chart, used for ROC curves. */
export function CurveChart({
  data,
  xKey,
  yKey,
  height = 260,
  diagonal,
}: {
  data: Record<string, any>[];
  xKey: string;
  yKey: string;
  height?: number;
  /** Draw the y=x reference line a ROC curve is read against. */
  diagonal?: boolean;
}) {
  const palette = usePalette();
  return (
    <ResponsiveContainer width="100%" height={height}>
      <LineChart data={data} margin={{ left: 0, right: 8, top: 8, bottom: 4 }}>
        <CartesianGrid stroke={palette.grid} />
        <XAxis dataKey={xKey} stroke={palette.text} {...AXIS_PROPS} />
        <YAxis stroke={palette.text} {...AXIS_PROPS} />
        <ChartTooltip palette={palette} />
        {diagonal && (
          <Line
            type="linear"
            dataKey="reference"
            stroke={palette.border}
            strokeDasharray="4 4"
            dot={false}
            isAnimationActive={false}
          />
        )}
        <Line
          type="monotone"
          dataKey={yKey}
          stroke={palette.primary}
          strokeWidth={2}
          dot={false}
        />
      </LineChart>
    </ResponsiveContainer>
  );
}

/** Scatter plot for actual-vs-predicted and residuals. */
export function ScatterPlot({
  data,
  xKey,
  yKey,
  height = 260,
  zeroLine,
}: {
  data: Record<string, any>[];
  xKey: string;
  yKey: string;
  height?: number;
  zeroLine?: boolean;
}) {
  const palette = usePalette();
  return (
    <ResponsiveContainer width="100%" height={height}>
      <ScatterChart margin={{ left: 0, right: 8, top: 8, bottom: 4 }}>
        <CartesianGrid stroke={palette.grid} />
        <XAxis type="number" dataKey={xKey} stroke={palette.text} {...AXIS_PROPS} />
        <YAxis type="number" dataKey={yKey} stroke={palette.text} {...AXIS_PROPS} />
        <ChartTooltip palette={palette} />
        {zeroLine && <ReferenceLine y={0} stroke={palette.action} strokeDasharray="4 4" />}
        <Scatter data={data} fill={palette.primary} fillOpacity={0.65} />
      </ScatterChart>
    </ResponsiveContainer>
  );
}

/** Confusion matrix rendered as a heat grid rather than a chart library plot. */
export function ConfusionMatrix({
  matrix,
  labels,
}: {
  matrix: number[][];
  labels: string[];
}) {
  const max = Math.max(1, ...matrix.flat());

  return (
    <div style={{ overflowX: 'auto' }}>
      <table className="data" style={{ tableLayout: 'fixed' }}>
        <thead>
          <tr>
            <th style={{ width: 110 }}>Actual \ Predicted</th>
            {labels.map((label) => (
              <th key={label} className="num">{label}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {matrix.map((row, i) => (
            <tr key={i}>
              <th scope="row">{labels[i]}</th>
              {row.map((value, j) => (
                <td
                  key={j}
                  className="num"
                  style={{
                    background: `color-mix(in srgb, var(--primary) ${(value / max) * 72}%, transparent)`,
                    fontWeight: i === j ? 600 : 400,
                  }}
                >
                  {value.toLocaleString()}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
