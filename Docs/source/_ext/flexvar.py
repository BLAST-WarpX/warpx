r"""
flexvar - A Sphinx domain for documenting variables with flexible names.

Supports names containing characters like <, >, /, commas, etc.
Provides type annotation, default value support, units, optional flag, and
extra inline comments. The output format can be changed by modifying
the `handle_signature` method in `FlexVarDirective`.

Usage
-----

[Directive](https://docutils.sourceforge.io/docs/ref/rst/directives.html)::

    .. fv:var:: <name>
        :type: <type>
        :default: <default>
        :unit: <unit>
        :optional:
        :comment: <comment>

        <Description body>

    Rendered format:

    <name>: (<type>; in <unit>) optional (default: <default>) <comment>

        <Description body>

[Cross reference role](https://www.sphinx-doc.org/en/master/usage/referencing.html)::

    :fv:var:`name`
    :fv:var:`Title <name>`
    :fv:var:`name = value`

Examples
--------

Directive::

    .. fv:var:: my/variable<T>
        :type: real array
        :default: [0.0, 0.0]
        :unit: seconds
        :optional:
        :comment: Inline comments on this variable.

        Description of the variable.

Cross reference role::

    # Cross reference link to my/variable<T>:
    See :fv:var:`my/variable<T>` for details.

    # With explicit title (whitespace required before the `<`):
    See :fv:var:`My Var <my/variable<T>>` for details.

    # With backslash-escaped angle brackets:
    See :fv:var:`my/variable\<T\>` for details.

    # With inline value:
    For instance, :fv:var:`my/variable<T> = [1, 1]`
"""

from __future__ import annotations

import re
from typing import Any, Iterator, TypedDict, cast

from docutils import nodes
from docutils.parsers.rst import directives
from sphinx import addnodes
from sphinx.addnodes import desc_signature
from sphinx.application import Sphinx
from sphinx.builders import Builder
from sphinx.directives import ObjectDescription
from sphinx.domains import Domain, ObjType
from sphinx.environment import BuildEnvironment
from sphinx.roles import XRefRole
from sphinx.util import logging
from sphinx.util.nodes import make_id, make_refnode

logger = logging.getLogger(__name__)


class ObjectEntry(TypedDict):
    docname: str
    node_id: str


class VarDesc:
    def __init__(self, sig: str):
        name_list: list[str] = sig.strip().split(" ")
        disp_name: str = name_list[0]
        if len(name_list) > 1:
            disp_name += ", ".join(name_list[1:-1]) + " & " + name_list[-1]

        self.display_name: str = disp_name
        self.name_list: list[str] = name_list
        self.sig: str = sig

    # def __str__(self):
    #     return self.display_name


class FlexVarDirective(ObjectDescription[str]):
    """
    Description of a variable.

    Supports variable names containing characters like <, >, /, commas, etc.
    See `handle_signature` for formatting details.

    Specify at most one of the following:

    - `default` and `value` are aliases
    - `unit` and `units` are aliases
    - `comment` and `annotation` are aliases
    - `optional` and `required` are opposite flags
    """

    option_spec = {
        "type": directives.unchanged,
        # `default` and `value` are aliases
        "default": directives.unchanged,
        "value": directives.unchanged,
        # `optional` and `required` flags are opposites. Do not specify both
        "optional": directives.flag,
        "required": directives.flag,
        # `unit` and `units` are aliases
        "unit": directives.unchanged,
        "units": directives.unchanged,
        # `comment` and `annotation` are aliases
        "comment": directives.unchanged,
        "annotation": directives.unchanged,
        "noindex": directives.flag,
    }

    # Disallow multiple names on a single directive line (commas in the name
    # would be mis-parsed otherwise).
    allow_nesting = False

    # Use emphasis for type and default value
    use_emphasis = False

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Equivalent names for physical units.
        # Aliases will be mapped to the first name of each list, e.g.,
        # ["m", "meter", "meters"] implies "meter" -> "m" and "meters" -> "m"
        unit_alias_lists: list[list[str]] = [
            ["m", "meter", "meters"],
            ["s", "second", "seconds"],
            ["kg", "kilogram", "kilograms"],
            ["V", "volt", "volts"],
            ["eV", "electronvolt", "electronvolts", "electron volt", "electron volts"],
            ["T", "Telsa", "Telsas"],
            ["A", "amp", "amps"],
            ["C", "Coulomb", "Coulombs"],
            ["dimensionless", "unitless", "none"],
        ]
        self.make_unit_alias_dict(unit_alias_lists)

        # type_alias_lists: list[list[str]] = [
        #     ["bool", "boolean"],
        #     ["int", "integer"],
        #     ["string", "str"],
        # ]

    def make_unit_alias_dict(self, unit_alias_lists: list[list[str]]):
        """
        Construct unit alias dict, e.g., "seconds" -> "s".

        Map aliases to first name of each list in `unit_alias_lists`, e.g.,
        ["m", "meter", "meters"] implies "meter" -> "m" and "meters" -> "m".
        """
        self.unit_alias_dict: dict[str, str] = {}
        for unit_alias_list in unit_alias_lists:
            for unit_alias in unit_alias_list[1:]:
                # Map aliases to the first name
                self.unit_alias_dict[unit_alias] = unit_alias_list[0]

    # ------------------------------------------------------------------
    # Signature parsing / rendering
    # ------------------------------------------------------------------

    def handle_signature(self, sig: str, signode: desc_signature) -> str:
        """
        Build the rendered signature node and return the canonical name.

        ``<name>: (<type>; in <unit>) [optional|required] (default: <default>) <comment>``
        """
        name = sig.strip()

        signode["fullname"] = name
        signode["ids"] = []  # filled in add_target_and_index

        # The variable name itself
        signode += addnodes.desc_name(
            name,
            "",
            self.parse_inline(name),
        )

        helper = FlexVarOptionHelper(
            options=self.options,
            name=name,
            signode=signode,
        )

        type_: str | None = helper.get_and_check_aliases("type")
        value: str | None = helper.get_and_check_aliases("default", "value")
        unit: str | None = helper.get_and_check_aliases("unit", "units")
        anno: str | None = helper.get_and_check_aliases("annotation", "comment")

        l_optional: bool = "optional" in self.options
        l_required: bool = "required" in self.options
        helper.check_conflicting_options("optional", "required")

        # Process unit string:
        if unit:
            unit = self.process_unit_string(unit)

        # Format: (`<type>`; in <unit>)
        if type_ or unit:
            # signode += addnodes.desc_sig_punctuation("", ":")
            signode += addnodes.desc_sig_space()
            signode += addnodes.desc_sig_punctuation("", "(")
            if type_:
                type_node = self.parse_inline(type_)
                if self.use_emphasis:
                    signode += nodes.emphasis(type_, "", type_node)
                else:
                    signode += type_node
            if type_ and unit:
                signode += addnodes.desc_sig_punctuation("", ";")
                signode += addnodes.desc_sig_space()
            if unit:
                if unit.lower() not in ["dimensionless", "unitless"]:
                    signode += nodes.Text("in")
                    signode += addnodes.desc_sig_space()
                signode += self.parse_inline(unit)
            signode += addnodes.desc_sig_punctuation("", ")")

        # Format: optional, required, or possibly neither
        if l_optional:
            signode += addnodes.desc_sig_space()
            signode += nodes.Text("optional")
        elif l_required:
            signode += addnodes.desc_sig_space()
            signode += nodes.Text("required")
        else:
            # Do nothing if neither flag is specified
            pass

        # Format: (default: `<default>`)
        if value:
            signode += addnodes.desc_sig_space()
            signode += addnodes.desc_sig_punctuation("", "(")
            signode += nodes.Text("default")
            signode += addnodes.desc_sig_punctuation("", ":")
            signode += addnodes.desc_sig_space()
            value_node = self.parse_inline(value)
            if self.use_emphasis:
                signode += nodes.emphasis(value, "", value_node)
            else:
                signode += value_node
            signode += addnodes.desc_sig_punctuation("", ")")

        # Format: <comment>
        if anno:
            signode += addnodes.desc_sig_space()
            signode += self.parse_inline(anno)

        return name

    def parse_inline(self, text: str) -> nodes.inline:
        """
        Parse *text* as RST inline content and return a single inline node.
        """
        parsed_nodes, messages = self.state.inline_text(text, self.lineno)
        # Report any parse warnings through the normal directive machinery
        for msg in messages:
            self.state_machine.reporter.system_message(
                msg["level"], msg.astext(), source=self.get_source_info()[0]
            )
        inline_node = nodes.inline(text, "", *parsed_nodes)
        return inline_node

    def process_unit_string(self, unit_str: str) -> str:
        result: str = unit_str
        # x / y -> x/y
        result = "/".join([u.strip() for u in result.split("/")])
        # x**n -> x^n
        result = result.replace("**", "^")
        # x.y -> x y
        result = result.replace(".", " ")
        # Use preferred alias for physical units
        for unit_alias, preferred_unit_name in self.unit_alias_dict.items():
            unit_re = re.compile(rf"\b{re.escape(unit_alias)}\b", re.IGNORECASE)
            result = re.sub(unit_re, preferred_unit_name, result)

        return result

    # ------------------------------------------------------------------
    # Index + target registration
    # ------------------------------------------------------------------

    def add_target_and_index(
        self, name: str, sig: str, signode: desc_signature
    ) -> None:
        node_id = make_id(self.env, self.state.document, "", name)
        signode["ids"].append(node_id)
        self.state.document.note_explicit_target(signode)

        domain = cast(FlexVarDomain, self.env.get_domain(FlexVarDomain.name))
        domain.note_var(
            name=name,
            docname=self.env.docname,
            node_id=node_id,
            location=signode,
        )

        if "noindex" not in self.options:
            self.indexnode["entries"].append(
                ("single", name + " (variable)", node_id, "", None)
            )


class FlexVarOptionHelper:
    def __init__(
        self,
        options: dict[str, Any],
        name: str,
        signode: desc_signature,
    ):
        self.options: dict[str, Any] = options
        self.name: str = name
        self.signode: desc_signature = signode

    def get_and_check_aliases(self, *keys: str, default=None) -> Any:
        if len(keys) > 1:
            self.check_conflicting_options(*keys)
        for key in keys:
            if key in self.options:
                return self.options[key]
        return default

    def check_conflicting_options(self, *keys: str):
        blist: list[bool] = [(key in self.options) for key in keys]
        if sum(blist) > 1:
            logger.warning(
                "Conflicting options for %s: specify only one of :%s",
                self.name,
                keys,
                location=self.signode,
            )


class FlexVarXRefRole(XRefRole):
    r"""
    Cross-referencing role for flexible name variables.

    Usage::

        :fv:var:`name`
        :fv:var:`Title <name>`
        :fv:var:`name = value`

    Customisations over the base ``XRefRole``:

    **Generic-style names**
        Variable names may contain ``<`` and ``>``
        (e.g. ``filter<T>``).  We require whitespace before the ``<`` that
        separates an explicit title from its target, so bare names like
        ``filter<T>`` are never mis-split.  This is done by overriding
        ``explicit_title_re``, which ``ReferenceRole.__call__`` uses directly.

    **Inline value syntax**
        A cross-reference may include a value
        expression after `` = ``. For example::

            :fv:var:`timeout = 30`

        will display as ``timeout = 30``. The value expression after
        is kept in the displayed title but stripped from the lookup target so
        that it still resolves to the ``.. fv:var:: timeout`` entry.
        Backslash-escape support (``\<``, ``\>``) comes for free from the
        base class.
    """

    # Same as ReferenceRole.explicit_title_re but with \s+ instead of \s*,
    # so whitespace before the `<` is required for explicit-title syntax.
    # \x00 means the "<" was backslash-escaped. Preserve that lookbehind.
    explicit_title_re = re.compile(r"^(.+?)\s+(?<!\x00)<(.*?)>$", re.DOTALL)

    # Matches an inline value expression: "varname = value" or "varname[=value]"
    # The name portion (before = or [=) is captured as group 1.
    _value_re = re.compile(r"^(.+?)(?:\s*=\s*.*|\[=.*\])$", re.DOTALL)

    def process_link(
        self,
        env: BuildEnvironment,
        refnode: nodes.Element,
        has_explicit_title: bool,
        title: str,
        target: str,
    ) -> tuple[str, str]:
        """Strip any inline value expression from the target, keeping it in the title."""
        if not has_explicit_title:
            m = self._value_re.match(target)
            if m:
                target = m.group(1).strip()
        return XRefRole.process_link(
            self,
            env=env,
            refnode=refnode,
            has_explicit_title=has_explicit_title,
            title=title,
            target=target,
        )


class FlexVarDomain(Domain):
    """FlexVar domain."""

    name = "fv"
    label = "FlexVar"

    object_types = {
        "var": ObjType("variable", "var"),
    }

    directives = {
        "var": FlexVarDirective,
    }

    roles = {
        "var": FlexVarXRefRole(),
    }

    initial_data: dict[str, dict[str, ObjectEntry]] = {
        "vars": {},
    }

    @property
    def vars(self) -> dict[str, ObjectEntry]:
        return self.data.setdefault("vars", {})

    def note_var(
        self,
        name: str,
        docname: str,
        node_id: str,
        location: Any = None,
    ) -> None:
        if name in self.vars:
            other = self.vars[name]
            logger.warning(
                "duplicate object description of %s, "
                "other instance in %s, use :noindex: for one of them",
                name,
                other["docname"],
                location=location,
            )

        self.vars[name] = {
            "docname": docname,
            "node_id": node_id,
        }

    def clear_doc(self, docname: str) -> None:
        to_remove = [k for k, v in self.vars.items() if v["docname"] == docname]
        for k in to_remove:
            del self.vars[k]

    def merge_domaindata(self, docnames: list[str], otherdata: dict[str, dict]) -> None:
        for name, info in otherdata.get("vars", {}).items():
            info = cast(ObjectEntry, info)
            if info["docname"] in docnames:
                self.vars[name] = info

    def resolve_xref(
        self,
        env: BuildEnvironment,
        fromdocname: str,
        builder: Builder,
        typ: str,
        target: str,
        node: addnodes.pending_xref,
        contnode: nodes.Element,
    ) -> nodes.Element | None:
        info = self.vars.get(target)
        if info is None:
            return None
        return make_refnode(
            builder,
            fromdocname,
            info["docname"],
            info["node_id"],
            contnode,
            target,
        )

    def get_objects(self) -> Iterator[tuple[str, str, str, str, str, int]]:
        for name, info in self.vars.items():
            yield (
                name,  # name
                name,  # dispname
                "var",  # type
                info["docname"],  # docname
                info["node_id"],  # anchor
                1,  # priority
            )


def setup(app: Sphinx) -> dict:
    app.add_domain(FlexVarDomain)
    return {
        "version": "0.1.0",
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }
