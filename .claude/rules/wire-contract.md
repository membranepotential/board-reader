---
paths:
  - 'packages/core/src/board_reader_core/protocol.py'
  - 'packages/core/src/board_reader_core/control.py'
  - 'packages/core/src/board_reader_core/wire.py'
  - 'packages/core/src/board_reader_core/wiremodel.py'
  - 'packages/core/src/board_reader_core/schema.py'
  - 'packages/core/tests/test_schema.py'
---

# The wire contract: pydantic models and their JSON Schema (DESIGN D13)

- Model a wire message as a pydantic dataclass through `wiremodel.wire` / `wiremodel.message`, never a `BaseModel`, because inference and edge build these positionally (`ConfigMessage(config)`, `CropRect(0.1, 0.1, 0.5, 0.5)`), compare them and pass them to `dataclasses.replace`, which a `BaseModel` does not take.
- Make scalars strict with pydantic's `Strict*` types (or `wiremodel`'s `Natural`, `Unit`, `BoardId`), never `ConfigDict(strict=True)`, because strict mode in Python mode refuses a dict for a nested dataclass and a list for a tuple, so `from_dict` on decoded JSON fails.
- Put a field alias or constraint in the annotation (`Annotated[ClassName, Field(alias="from")]`), never as the default (`= Field(alias="from")`), because the standard dataclass machinery reads a `Field(...)` default as a default and then refuses the next field without one.
- Round a float with `AfterValidator(rounded(n))`, never a `PlainSerializer`, because a plain serializer replaces the field's serialization schema and the range (`minimum`/`maximum`) drops out of the exported JSON Schema.
- Give a message its `type` with `tag(...)` (keyword-only, defaulted) and let `message` derive `TYPE` from it, because the keyword-only tag keeps the message's own fields positional, and the `Message` base's before-validator still requires `type` when the message is read off a wire.
- Bound every wire integer by `SAFE_INT_MAX` and write regex classes as `[0-9]`, never `\d`, because zod's `z.int()` refuses integers past 2**53 - 1, and `\d` matches Unicode digits in Python and in pydantic's Rust regex but only ASCII digits in JavaScript.
- State a rule JSON Schema cannot express (two fields together, a FEN rank's length) as a validator plus `checked("<name>")`, and add the name to `test_the_rules_no_keyword_states_are_named_checks`, because the TypeScript side implements each named check by hand and the name is the only link between the two.
- Add every new browser-facing message to `schema._roots()`, because only those reach the TypeScript generator; a model left out still validates in Python and silently has no TypeScript twin.
