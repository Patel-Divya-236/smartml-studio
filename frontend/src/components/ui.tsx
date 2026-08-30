import {
  useEffect,
  useState,
  type ButtonHTMLAttributes,
  type ReactNode,
} from 'react';
import './ui.css';

/* ── Page scaffolding ─────────────────────────────────────────────── */

export function Page({
  title,
  subtitle,
  actions,
  children,
}: {
  title: string;
  subtitle?: string;
  actions?: ReactNode;
  children: ReactNode;
}) {
  return (
    <div className="page">
      <header className="page-header row-between">
        <div>
          <h1 className="page-title">{title}</h1>
          {subtitle && <p className="page-subtitle">{subtitle}</p>}
        </div>
        {actions && <div className="row">{actions}</div>}
      </header>
      {children}
    </div>
  );
}

export function Card({
  title,
  hint,
  actions,
  className = '',
  children,
}: {
  title?: string;
  hint?: string;
  actions?: ReactNode;
  className?: string;
  children: ReactNode;
}) {
  return (
    <section className={`card ${className}`}>
      {(title || actions) && (
        <div className="card-header">
          <div>
            {title && <h2 className="card-title">{title}</h2>}
            {hint && <p className="card-hint">{hint}</p>}
          </div>
          {actions}
        </div>
      )}
      {children}
    </section>
  );
}

/* ── Button ───────────────────────────────────────────────────────── */

type ButtonVariant = 'primary' | 'action' | 'secondary' | 'ghost';

export function Button({
  variant = 'secondary',
  size = 'md',
  block,
  loading,
  children,
  className = '',
  ...rest
}: ButtonHTMLAttributes<HTMLButtonElement> & {
  variant?: ButtonVariant;
  size?: 'sm' | 'md' | 'lg';
  block?: boolean;
  loading?: boolean;
}) {
  const classes = [
    'btn',
    `btn-${variant}`,
    size !== 'md' ? `btn-${size}` : '',
    block ? 'btn-block' : '',
    className,
  ]
    .filter(Boolean)
    .join(' ');

  return (
    <button className={classes} disabled={rest.disabled || loading} {...rest}>
      {loading && <span className="spinner" />}
      {children}
    </button>
  );
}

/* ── Badge ────────────────────────────────────────────────────────── */

export type BadgeTone =
  | 'neutral'
  | 'primary'
  | 'accent'
  | 'success'
  | 'warning'
  | 'danger'
  | 'info';

export function Badge({ tone = 'neutral', children }: { tone?: BadgeTone; children: ReactNode }) {
  return <span className={`badge badge-${tone}`}>{children}</span>;
}

/* ── Stat tile ────────────────────────────────────────────────────── */

export function Stat({
  value,
  label,
  sub,
  icon,
}: {
  value: ReactNode;
  label: string;
  sub?: string;
  icon?: ReactNode;
}) {
  return (
    <div className="stat">
      {icon && <div className="stat-icon">{icon}</div>}
      <div className="stat-value">{value}</div>
      <div className="stat-label">{label}</div>
      {sub && <div className="stat-sub">{sub}</div>}
    </div>
  );
}

/* ── Tabs ─────────────────────────────────────────────────────────── */

export function Tabs<T extends string>({
  tabs,
  active,
  onChange,
}: {
  tabs: { id: T; label: string; count?: number }[];
  active: T;
  onChange: (id: T) => void;
}) {
  return (
    <div className="tabs" role="tablist">
      {tabs.map((tab) => (
        <button
          key={tab.id}
          role="tab"
          className="tab"
          aria-selected={tab.id === active}
          onClick={() => onChange(tab.id)}
        >
          {tab.label}
          {tab.count !== undefined && ` (${tab.count})`}
        </button>
      ))}
    </div>
  );
}

/* ── Feedback ─────────────────────────────────────────────────────── */

export function Alert({
  tone = 'info',
  children,
}: {
  tone?: 'info' | 'success' | 'warning' | 'danger';
  children: ReactNode;
}) {
  return (
    <div className={`alert alert-${tone}`}>
      <span className="alert-dot" style={{ background: `var(--${tone})` }} />
      <div>{children}</div>
    </div>
  );
}

export function EmptyState({
  title,
  description,
  action,
}: {
  title: string;
  description?: string;
  action?: ReactNode;
}) {
  return (
    <div className="empty">
      <div className="empty-title">{title}</div>
      {description && <p className="small" style={{ maxWidth: '46ch' }}>{description}</p>}
      {action}
    </div>
  );
}

export function Progress({ value, total }: { value: number; total: number }) {
  const pct = total > 0 ? Math.min(100, (value / total) * 100) : 0;
  return (
    <div className="progress" role="progressbar" aria-valuenow={value} aria-valuemax={total}>
      <div className="progress-bar" style={{ width: `${pct}%` }} />
    </div>
  );
}

export function Loading({ label = 'Loading…' }: { label?: string }) {
  return (
    <div className="row muted" style={{ padding: 'var(--space-6)' }}>
      <span className="spinner" />
      <span className="small">{label}</span>
    </div>
  );
}

/* ── Tooltip ──────────────────────────────────────────────────────── */

export function Tooltip({ text, children }: { text: string; children: ReactNode }) {
  const [open, setOpen] = useState(false);
  return (
    <span
      className="tip"
      onMouseEnter={() => setOpen(true)}
      onMouseLeave={() => setOpen(false)}
      onFocus={() => setOpen(true)}
      onBlur={() => setOpen(false)}
    >
      {children}
      {open && <span className="tip-bubble" role="tooltip">{text}</span>}
    </span>
  );
}

/* ── Modal ────────────────────────────────────────────────────────── */

export function Modal({
  title,
  onClose,
  children,
}: {
  title: string;
  onClose: () => void;
  children: ReactNode;
}) {
  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      if (event.key === 'Escape') onClose();
    };
    document.addEventListener('keydown', onKey);
    return () => document.removeEventListener('keydown', onKey);
  }, [onClose]);

  return (
    <div className="overlay" onClick={onClose} role="presentation">
      <div
        className="modal"
        role="dialog"
        aria-modal="true"
        aria-label={title}
        onClick={(event) => event.stopPropagation()}
      >
        <div className="modal-header">
          <h2 className="card-title">{title}</h2>
          <Button variant="ghost" size="sm" onClick={onClose} aria-label="Close">
            ✕
          </Button>
        </div>
        <div className="modal-body">{children}</div>
      </div>
    </div>
  );
}

/* ── Form controls ────────────────────────────────────────────────── */

export function Field({ label, hint, children }: { label: string; hint?: string; children: ReactNode }) {
  return (
    <div className="field">
      <label className="label">{label}</label>
      {children}
      {hint && <span className="xs muted">{hint}</span>}
    </div>
  );
}

export function Select({
  value,
  options,
  onChange,
  small,
  ...rest
}: {
  value: string;
  options: string[];
  onChange: (value: string) => void;
  small?: boolean;
  disabled?: boolean;
  'aria-label'?: string;
}) {
  return (
    <select
      className={`select ${small ? 'select-sm' : ''}`}
      value={value}
      onChange={(event) => onChange(event.target.value)}
      {...rest}
    >
      {options.map((option) => (
        <option key={option} value={option}>
          {option}
        </option>
      ))}
    </select>
  );
}

export function Switch({
  checked,
  onChange,
  label,
}: {
  checked: boolean;
  onChange: (value: boolean) => void;
  label?: string;
}) {
  return (
    <div className="row">
      <button
        type="button"
        className="switch"
        role="switch"
        aria-checked={checked}
        aria-label={label}
        onClick={() => onChange(!checked)}
      />
      {label && <span className="small secondary">{label}</span>}
    </div>
  );
}

/* ── Table ────────────────────────────────────────────────────────── */

export interface Column<T> {
  key: string;
  header: string;
  numeric?: boolean;
  render?: (row: T) => ReactNode;
}

export function DataTable<T extends Record<string, any>>({
  columns,
  rows,
  rowClassName,
  emptyLabel = 'Nothing to show yet.',
  maxHeight,
}: {
  columns: Column<T>[];
  rows: T[];
  rowClassName?: (row: T) => string;
  emptyLabel?: string;
  maxHeight?: number;
}) {
  if (!rows.length) {
    return <p className="small muted" style={{ padding: 'var(--space-4)' }}>{emptyLabel}</p>;
  }

  return (
    <div className="table-wrap" style={maxHeight ? { maxHeight, overflowY: 'auto' } : undefined}>
      <table className="data">
        <thead>
          <tr>
            {columns.map((column) => (
              <th key={column.key} className={column.numeric ? 'num' : undefined}>
                {column.header}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row, index) => (
            <tr key={index} className={rowClassName?.(row)}>
              {columns.map((column) => (
                <td key={column.key} className={column.numeric ? 'num' : undefined}>
                  {column.render ? column.render(row) : formatCell(row[column.key])}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

/** Render a cell without inventing precision: integers stay whole, floats get 4 places. */
export function formatCell(value: unknown): ReactNode {
  if (value === null || value === undefined) return <span className="muted">—</span>;
  if (typeof value === 'boolean') return value ? 'Yes' : 'No';
  if (typeof value === 'number') {
    if (Number.isInteger(value)) return value.toLocaleString();
    return value.toFixed(4);
  }
  return String(value);
}

/* ── Confidence rating ────────────────────────────────────────────── */

export function ConfidenceStars({ rating, score }: { rating: number; score: number }) {
  return (
    <Tooltip text={`Rule confidence: ${(score * 100).toFixed(0)}%`}>
      <span className="row" style={{ gap: 2 }}>
        <span style={{ color: 'var(--accent)', letterSpacing: '1px', fontSize: 'var(--text-sm)' }}>
          {'★'.repeat(rating)}
          <span className="muted">{'★'.repeat(Math.max(0, 5 - rating))}</span>
        </span>
      </span>
    </Tooltip>
  );
}
