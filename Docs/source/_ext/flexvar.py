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

    <name> (<type>; in <unit>) optional (default: <default>) <comment>

        <Description body>

[Cross reference role](https://www.sphinx-doc.org/en/master/usage/referencing.html)::

    :fv:var:`name`
    :fv:var:`Title <name>`
    :fv:var:`name = value`

Examples
--------

Directive::

    .. fv:var:: my/variable<T>
        :link_aliases: my<T> variable<T>
        :type: real array
        :default: [0.0, 0.0]
        :unit: seconds
        :optional:
        :comment: Inline comments on this variable.

        Description of the variable.

Cross reference role::

    # Cross reference link to my/variable<T>:
    See :fv:var:`my/variable<T>` for details.

    # Alternative names also work as links:
    See :fv:var:`my<T>` or :fv:var:`variable<T>` for details.

    # With explicit title (whitespace required before the `<`):
    See :fv:var:`My Var <my/variable<T>>` for details.

    # With backslash-escaped angle brackets:
    See :fv:var:`my/variable\<T\>` for details.

    # With inline value:
    For instance, :fv:var:`my/variable<T> = [1, 1]`
"""

from __future__ import annotations

import re
from typing import Any, Iterator, NamedTuple, cast

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

type_equiv_name_dict: dict[str, list[str]] = {
    "bool": ["boolean", "logical"],
    "int": ["integer"],
    "string": ["str"],
}

unit_equiv_name_dict: dict[str, list[str]] = {
    "m": ["meter", "meters"],
    "s": ["second", "seconds"],
    "kg": ["kilogram", "kilograms"],
    "V": ["volt", "volts"],
    "eV": ["electronvolt", "electronvolts", "electron volt", "electron volts"],
    "T": ["Telsa", "Telsas"],
    "A": ["amp", "amps"],
    "C": ["Coulomb", "Coulombs"],
    "dimensionless": ["unitless", "none"],
}


def replace_equiv_names(txt: str, equiv_dict: dict[str, list[str]]) -> str:
    """Replace equivalent names in ``txt``, e.g., "seconds" -> "s".

    Each key, value in ``equiv_dict`` is a preferred name and equivalent name
    list, respectively. Replace each equivalent name to the preferred name,
    e.g., "m": ["meter", "meters"] implies "meter" -> "m" and "meters" -> "m".
    """
    preferred_name: str
    other_name_list: list[str]
    for preferred_name, other_name_list in equiv_dict.items():
        for other_name in other_name_list:
            # Only replace whole word matches. \b = word break.
            sub_re = re.compile(rf"\b{other_name}\b", re.IGNORECASE)
            txt = sub_re.sub(preferred_name, txt)
    return txt


class ObjectEntry(NamedTuple):
    docname: str
    node_id: str
    var_desc: VarDesc


class VarDesc(NamedTuple):
    display_name: str
    name_list: list[str]


class FlexVarDirective(ObjectDescription[VarDesc]):
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
        "link_aliases": directives.unchanged,
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

    # ------------------------------------------------------------------
    # Signature parsing / rendering
    # ------------------------------------------------------------------

    def handle_signature(self, sig: str, signode: desc_signature) -> VarDesc:
        """
        Build the rendered signature node and return a description object.

        ``<name> (<type>; in <unit>) [optional|required] (default: <default>) <comment>``
        """
        # Parse self.options
        helper = FlexVarOptionHelper(
            options=self.options,
            sig=sig,
            signode=signode,
        )
        link_aliases: str | None = helper.get_and_check_aliases("link_aliases")
        type_: str | None = helper.get_and_check_aliases("type")
        default: str | None = helper.get_and_check_aliases("default", "value")
        unit: str | None = helper.get_and_check_aliases("unit", "units")
        comment: str | None = helper.get_and_check_aliases("annotation", "comment")
        l_optional: bool = "optional" in self.options
        l_required: bool = "required" in self.options
        helper.check_conflicting_options("optional", "required")

        # Make canonical name and alternative name list
        name: str = re.sub(r"\s+", "", sig)

        alias_list: list[str] = link_aliases.split() if link_aliases else []

        signode["fullname"] = name
        signode["ids"] = []  # filled in add_target_and_index

        # Process strings:
        if type_:
            type_ = self._process_type_string(type_)
        if unit:
            unit = self._process_unit_string(unit)

        # Format: <name>
        signode += addnodes.desc_name(name, "", self.parse_inline(name))

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
        if type_ or unit:
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
        if default:
            signode += addnodes.desc_sig_space()
            signode += addnodes.desc_sig_punctuation("", "(")
            signode += nodes.Text("default")
            signode += addnodes.desc_sig_punctuation("", ":")
            signode += addnodes.desc_sig_space()
            value_node = self.parse_inline(default)
            if self.use_emphasis:
                signode += nodes.emphasis(default, "", value_node)
            else:
                signode += value_node
            signode += addnodes.desc_sig_punctuation("", ")")

        # Format: <comment>
        if comment:
            signode += addnodes.desc_sig_space()
            signode += self.parse_inline(comment)

        return VarDesc(name, alias_list)

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

    def _process_type_string(self, type_str: str) -> str:
        # Replace equivalent names for types, e.g., "integer" -> "int"
        type_str = replace_equiv_names(type_str, type_equiv_name_dict)
        return type_str

    def _process_unit_string(self, unit_str: str) -> str:
        # x / y -> x/y
        unit_str = "/".join([u.strip() for u in unit_str.split("/")])
        # x**n -> x^n
        unit_str = unit_str.replace("**", "^")
        # x.y -> x y
        unit_str = unit_str.replace(".", " ")
        # Replace equivalent names for physical units, e.g., "meter" -> "m"
        unit_str = replace_equiv_names(unit_str, unit_equiv_name_dict)
        return unit_str

    # ------------------------------------------------------------------
    # Index + target registration
    # ------------------------------------------------------------------

    def add_target_and_index(
        self, var_desc: VarDesc, sig: str, signode: desc_signature
    ) -> None:
        name: str = var_desc.display_name
        node_id = make_id(self.env, self.state.document, "", name)
        signode["ids"].append(node_id)
        self.state.document.note_explicit_target(signode)

        domain = cast(FlexVarDomain, self.env.get_domain(FlexVarDomain.name))
        domain.note_object(
            var_desc=var_desc,
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
        sig: str,
        signode: desc_signature,
    ):
        self.options: dict[str, Any] = options
        self.sig: str = sig
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
                self.sig.strip(),
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

    Customizations over the base ``XRefRole``:

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
            self, env, refnode, has_explicit_title, title, target
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
        "objects": {},
    }

    @property
    def objects(self) -> dict[str, ObjectEntry]:
        return self.data.setdefault("objects", {})

    def note_object(
        self,
        var_desc: VarDesc,
        node_id: str,
        location: Any = None,
    ) -> None:
        name = var_desc.display_name
        if name in self.objects:
            other = self.objects[name]
            logger.warning(
                "duplicate object description of %s, "
                "other instance in %s, use :noindex: for one of them",
                name,
                other.docname,
                location=location,
            )

        self.objects[name] = ObjectEntry(
            docname=self.env.docname,
            node_id=node_id,
            var_desc=var_desc,
        )

        for alt_name in var_desc.name_list:
            self.objects[alt_name] = self.objects[name]

    def clear_doc(self, docname: str) -> None:
        to_remove = [k for k, v in self.objects.items() if v.docname == docname]
        for k in to_remove:
            del self.objects[k]

    def merge_domaindata(self, docnames: list[str], otherdata: dict[str, dict]) -> None:
        for name, info in otherdata.get("objects", {}).items():
            info = cast(ObjectEntry, info)
            if info.docname in docnames:
                self.objects[name] = info

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
        obj: ObjectEntry | None = self.objects.get(target)
        if obj is None:
            return None
        return make_refnode(
            builder=builder,
            fromdocname=fromdocname,
            todocname=obj.docname,
            targetid=obj.node_id,
            child=contnode,
            title=target,
        )

    def get_objects(self) -> Iterator[tuple[str, str, str, str, str, int]]:
        """Return an iterable of "object descriptions".

        Object descriptions are tuples with six items (from Domain.get_objects
        docstring):

        ``name``
          Fully qualified name.

        ``dispname``
          Name to display when searching/linking.

        ``type``
          Object type, a key in ``self.object_types``.

        ``docname``
          The document where it is to be found.

        ``anchor``
          The anchor name for the object.

        ``priority``
          How "important" the object is (determines placement in search
          results). One of:

          ``1``
            Default priority (placed before full-text matches).
          ``0``
            Object is important (placed before default-priority objects).
          ``2``
            Object is unimportant (placed after full-text matches).
          ``-1``
            Object should not show up in search at all.
        """
        for obj_key, obj in self.objects.items():
            name: str = obj_key
            dispname: str = obj.var_desc.display_name
            type_: str = "var"
            docname: str = obj.docname
            anchor: str = obj.node_id
            priority: int = 0
            yield (name, dispname, type_, docname, anchor, priority)


class WarpXDomain(FlexVarDomain):
    label = "WarpX"

    object_types = {
        "var": ObjType("parameter", "var"),
    }


def setup(app: Sphinx) -> dict:
    app.add_domain(WarpXDomain)
    return {
        "version": "0.1.0",
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }
