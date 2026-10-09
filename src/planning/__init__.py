"""LLM planning layer: the planner proposes, a deterministic validator decides.

    profile ─► profile_digest ─► planner (LLM, JSON) ─► schema (shape) ─► validator (rules)
                                                                              │
                     rule-based advisors ─────────────────────► merge ◄───────┘
                                                                  │
                                                         user approves / edits
"""
