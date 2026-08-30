import { useEffect, useState, type ReactNode } from 'react';
import { STEPS, STEP_GROUPS, isUnlocked, usePipeline } from '../store/pipeline';
import { useTheme } from '../theme/ThemeProvider';
import { resetSession } from '../api/client';
import { Switch, Tooltip } from './ui';
import './shell.css';

/**
 * Application shell: a persistent rail on the left, one module in the main area.
 *
 * The rail replaces Streamlit's flat page list. It shows the pipeline as the dependency
 * chain it actually is — completed steps carry a tick, steps whose prerequisites are
 * unmet are disabled with the reason in a tooltip — so a user can see where they are
 * without reading a warning after clicking.
 */
export function Shell({ children }: { children: ReactNode }) {
  const { currentStep, completed, goTo, meta, autoAdvance, setAutoAdvance, refreshState } =
    usePipeline();
  const { theme, toggle } = useTheme();
  const [navOpen, setNavOpen] = useState(false);

  useEffect(() => {
    void refreshState();
  }, [refreshState]);

  // Close the mobile drawer whenever the module changes.
  useEffect(() => setNavOpen(false), [currentStep]);

  const activeStep = STEPS.find((step) => step.id === currentStep);
  const doneCount = Object.values(completed).filter(Boolean).length;

  function startOver() {
    if (!window.confirm('Discard this session and start a new pipeline?')) return;
    resetSession();
    usePipeline.getState().reset();
    void refreshState();
  }

  return (
    <div className="shell">
      <aside className={`sidebar ${navOpen ? 'is-open' : ''}`}>
        <div className="brand">
          <span className="brand-mark" aria-hidden="true" />
          <span className="brand-name">SmartML Studio</span>
        </div>

        <nav className="nav" aria-label="Pipeline">
          {STEP_GROUPS.map((group) => {
            const steps = STEPS.filter((step) => step.group === group);
            if (!steps.length) return null;
            return (
              <div className="nav-group" key={group}>
                <div className="nav-group-label">{group}</div>
                {steps.map((step) => {
                  const unlocked = isUnlocked(step, completed);
                  const done = completed[step.id];
                  const blocker = step.requires.find((required) => !completed[required]);
                  const blockerLabel = STEPS.find((s) => s.id === blocker)?.label;

                  const item = (
                    <button
                      key={step.id}
                      className="nav-item"
                      aria-current={step.id === currentStep ? 'page' : undefined}
                      data-locked={!unlocked}
                      disabled={!unlocked}
                      onClick={() => goTo(step.id)}
                    >
                      <span className="nav-dot" data-done={done} aria-hidden="true" />
                      <span className="nav-label">{step.label}</span>
                      {!unlocked && <span className="nav-lock" aria-hidden="true">🔒</span>}
                    </button>
                  );

                  return unlocked ? (
                    item
                  ) : (
                    <Tooltip key={step.id} text={`Complete ${blockerLabel} first.`}>
                      {item}
                    </Tooltip>
                  );
                })}
              </div>
            );
          })}
        </nav>

        <div className="sidebar-footer">
          <div className="progress-line">
            <span className="xs muted">{doneCount} of {STEPS.length} steps complete</span>
          </div>
          <Switch checked={autoAdvance} onChange={setAutoAdvance} label="Auto-advance" />
          <span className="xs muted">
            Moves on after steps with no result to read. Continue buttons always work.
          </span>
          <button className="link-btn" onClick={startOver}>
            Start over
          </button>
        </div>
      </aside>

      {navOpen && <div className="nav-scrim" onClick={() => setNavOpen(false)} role="presentation" />}

      <div className="main">
        <header className="topbar">
          <button
            className="icon-btn nav-toggle"
            onClick={() => setNavOpen((open) => !open)}
            aria-label="Toggle navigation"
          >
            ☰
          </button>

          <div className="crumbs">
            <span className="muted">{activeStep?.group}</span>
            <span className="muted" aria-hidden="true">/</span>
            <span className="strong">{activeStep?.label}</span>
          </div>

          <div className="spacer" />

          {meta.dataset_name && (
            <div className="context">
              <span className="context-item truncate" title={meta.dataset_name}>
                {meta.dataset_name}
              </span>
              {meta.target_column && (
                <span className="context-item">
                  target: <span className="strong">{meta.target_column}</span>
                </span>
              )}
              {meta.problem_type && <span className="context-item">{meta.problem_type}</span>}
            </div>
          )}

          <Tooltip text={theme === 'dark' ? 'Switch to light theme' : 'Switch to dark theme'}>
            <button className="icon-btn" onClick={toggle} aria-label="Toggle theme">
              {theme === 'dark' ? '☀' : '☾'}
            </button>
          </Tooltip>
        </header>

        <main className="content">{children}</main>
      </div>
    </div>
  );
}
