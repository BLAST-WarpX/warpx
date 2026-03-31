r"""parmparse - A Sphinx domain for documenting ParamParse parameters.

Supports names containing characters like <, >, /, commas, etc.
Provides type annotation, default value support, units, optional flag, and
extra inline comments. The output format can be changed by modifying
the `handle_signature` method in `ParmParseDirective`.

Usage
-----

[Directive](https://docutils.sourceforge.io/docs/ref/rst/directives.html)::

    .. pp:param:: <name>
        :type: <type>
        :default: <default>
        :unit: <unit>
        :optional:
        :comment: <comment>

        <Description body>

    Rendered format:

    <name> (<type>; [<unit>]; optional, default: <default>) <comment>

        <Description body>

[Cross reference role](https://www.sphinx-doc.org/en/master/usage/referencing.html)::

    :pp:param:`name`
    :pp:param:`Title <name>`
    :pp:param:`name = value`

Examples
--------

Directive::

    .. pp:param:: my/parameter<T>
        :link_aliases: my<T> parameter<T>
        :type: real array
        :default: [0.0, 0.0]
        :unit: seconds
        :optional:
        :comment: Inline comments on this parameter.

        Description of the parameter.

Cross reference role::

    # Cross reference link to my/parameter<T>:
    See :pp:param:`my/parameter<T>` for details.

    # Alternative names also work as links:
    See :pp:param:`my<T>` or :pp:param:`parameter<T>` for details.

    # With explicit title (whitespace required before the `<`):
    See :pp:param:`My param <my/parameter<T>>` for details.

    # With backslash-escaped angle brackets:
    See :pp:param:`my/parameter\<T\>` for details.

    # With inline value:
    For instance, :pp:param:`my/parameter<T> = [1, 1]`
"""

from __future__ import annotations

import re
from typing import Any, Iterator, Literal, NamedTuple, cast

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
    "speed of light fraction": ["units of the speed of light"],
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
    desc: ObjDesc


class ObjDesc(NamedTuple):
    directive_target_name: str
    link_aliases: list[str]


class ParmParseDirective(ObjectDescription[ObjDesc]):
    """
    Description of a parameter.

    Supports parameter names containing characters like <, >, /, commas, etc.
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

    def handle_signature(self, sig: str, signode: desc_signature) -> ObjDesc:
        """
        Build the rendered signature node and return a description object.

        ``<name> (<type>; [<unit>]; optional, default: <default>) <comment>``
        """
        # Parse self.options
        helper = ParmParseOptionHelper(
            options=self.options,
            sig=sig,
            signode=signode,
        )
        aliases_str: str | None = helper.get_option("link_aliases")
        type_str: str | None = helper.get_option("type")
        default_str: str | None = helper.get_option("default", "value")
        unit_str: str | None = helper.get_option("unit", "units")
        comment_str: str | None = helper.get_option("annotation", "comment")
        l_optional: bool = "optional" in self.options
        l_required: bool = "required" in self.options
        helper.check_conflicting_options("optional", "required")

        # Make canonical name
        name: str = re.sub(r"\s+", "", sig)

        # Make alternative name list
        alias_list: list[str] = aliases_str.split() if aliases_str else []

        signode["fullname"] = name
        signode["ids"] = []  # filled in add_target_and_index

        # Process strings:
        if type_str:
            type_str = self._process_type_string(type_str)
        if unit_str:
            unit_str = self._process_unit_string(unit_str)

        # Format: <name>
        signode += addnodes.desc_name(name, "", self.parse_inline(name))

        # Format: (<type>; [<unit>]; optional, default: <default>)
        if type_str or unit_str or l_optional or l_required or default_str:
            signode += addnodes.desc_sig_space()
            signode += addnodes.desc_sig_punctuation("", "(")
        # Format: <type>;
        if type_str:
            type_node = self.parse_inline(type_str)
            if self.use_emphasis:
                signode += nodes.emphasis(type_str, "", type_node)
            else:
                signode += type_node
        if type_str and (unit_str or l_optional or l_required or default_str):
            signode += addnodes.desc_sig_punctuation("", ";")
            signode += addnodes.desc_sig_space()
        # Format: [<unit>];
        if unit_str:
            signode += addnodes.desc_sig_punctuation("", "[")
            signode += self.parse_inline(unit_str)
            signode += addnodes.desc_sig_punctuation("", "]")
        if unit_str and (l_optional or l_required or default_str):
            signode += addnodes.desc_sig_punctuation("", ";")
            signode += addnodes.desc_sig_space()
        # Format: optional, required, or possibly neither
        if l_optional:
            signode += nodes.Text("optional")
        elif l_required:
            signode += nodes.Text("required")
        else:
            # Do nothing if neither flag is specified
            pass
        if (l_optional or l_required) and default_str:
            signode += addnodes.desc_sig_punctuation("", ",")
            signode += addnodes.desc_sig_space()
        # default: <default>
        if default_str:
            signode += nodes.Text("default")
            signode += addnodes.desc_sig_punctuation("", ":")
            signode += addnodes.desc_sig_space()
            default_node = self.parse_inline(default_str)
            if self.use_emphasis:
                signode += nodes.emphasis(default_str, "", default_node)
            else:
                signode += default_node
        if type_str or unit_str or l_optional or l_required or default_str:
            signode += addnodes.desc_sig_punctuation("", ")")

        # Format: <comment>
        if comment_str:
            signode += addnodes.desc_sig_space()
            signode += self.parse_inline(comment_str)

        return ObjDesc(name, alias_list)

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
        # Replace equivalent names for physical units, e.g., "meter" -> "m"
        unit_str = replace_equiv_names(unit_str, unit_equiv_name_dict)
        # (x / y or x per y) -> x/y
        unit_str = re.sub(r"\b(\s*/\s*|\s+per\s+)\b", "/", unit_str)
        # x**n or x^n -> x\ :sup:`n`
        superscript_pattern = re.compile(r"\b(?:\*\*|\^)(\w+)\b")
        def superscript_repl(m: re.Match):
            # Tip from https://docutils.sourceforge.io/docs/ref/rst/roles.html#subscript
            return r"\ :sup:`" + m.group(1) + r"`"
        unit_str = superscript_pattern.sub(superscript_repl, unit_str)
        # x.y -> x y
        unit_str = re.sub(r"\b\.\b", " ", unit_str)
        return unit_str

    # ------------------------------------------------------------------
    # Index + target registration
    # ------------------------------------------------------------------

    def add_target_and_index(
        self, desc: ObjDesc, sig: str, signode: desc_signature
    ) -> None:
        name: str = desc.directive_target_name
        node_id = make_id(self.env, self.state.document, "", name)
        signode["ids"].append(node_id)
        self.state.document.note_explicit_target(signode)

        domain = cast(ParmParseDomain, self.env.get_domain(ParmParseDomain.name))
        domain.note_object(
            desc=desc,
            node_id=node_id,
            location=signode,
        )

        if "noindex" not in self.options:
            self.indexnode["entries"].append(
                ("single", name + " (parameter)", node_id, "", None)
            )


class ParmParseOptionHelper:
    def __init__(
        self,
        options: dict[str, Any],
        sig: str,
        signode: desc_signature,
    ):
        self.options: dict[str, Any] = options
        self.sig: str = sig
        self.signode: desc_signature = signode

    def get_option(self, *keys: str, default=None) -> Any:
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


class ParmParseXRefRole(XRefRole):
    r"""
    Cross-referencing role for flexible name parameters.

    Usage::

        :pp:param:`name`
        :pp:param:`Title <name>`
        :pp:param:`name = value`

    Customizations over the base ``XRefRole``:

    **Generic-style names**
        parameter names may contain ``<`` and ``>``
        (e.g. ``filter<T>``).  We require whitespace before the ``<`` that
        separates an explicit title from its target, so bare names like
        ``filter<T>`` are never mis-split.  This is done by overriding
        ``explicit_title_re``, which ``ReferenceRole.__call__`` uses directly.

    **Inline value syntax**
        A cross-reference may include a value
        expression after `` = ``. For example::

            :pp:param:`timeout = 30`

        will display as ``timeout = 30``. The value expression after
        is kept in the displayed title but stripped from the lookup target so
        that it still resolves to the ``.. pp:param:: timeout`` entry.
        Backslash-escape support (``\<``, ``\>``) comes for free from the
        base class.
    """

    # Same as ReferenceRole.explicit_title_re but with \s+ instead of \s*,
    # so whitespace before the `<` is required for explicit-title syntax.
    # \x00 means the "<" was backslash-escaped. Preserve that lookbehind.
    explicit_title_re = re.compile(r"^(.+?)\s+(?<!\x00)<(.*?)>$", re.DOTALL)

    # Matches an inline value expression.
    # Examples: "param_name = value", "param_name[=value]", etc.
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

        refnode["refdomain"] = ParmParseDomain.name
        refnode["reftype"] = "param"

        return XRefRole.process_link(
            self, env, refnode, has_explicit_title, title, target
        )


class ParmParseDomain(Domain):
    """ParmParse domain."""

    name = "pp"
    label = "ParmParse"

    object_types = {
        "param": ObjType("parameter", "param"),
    }

    directives = {
        "param": ParmParseDirective,
    }

    roles = {
        "param": ParmParseXRefRole(),
    }

    initial_data: dict[str, dict[str, ObjectEntry]] = {
        "objects": {},
    }

    @property
    def objects(self) -> dict[str, ObjectEntry]:
        return self.data.setdefault("objects", {})

    def note_object(
        self,
        desc: ObjDesc,
        node_id: str,
        location: Any = None,
    ) -> None:
        name = desc.directive_target_name
        obj = ObjectEntry(
            docname=self.env.docname,
            node_id=node_id,
            desc=desc,
        )
        self.add_object(name, obj, location)

        for alt_name in desc.link_aliases:
            self.add_object(alt_name, obj, location)

    def add_object(self, name: str, obj: ObjectEntry, location: Any) -> None:
        if name in self.objects:
            other = self.objects[name]
            logger.warning(
                "duplicate object description of %s, "
                "other instance in %s, use :noindex: for one of them",
                name,
                other.docname,
                location=location,
            )
        self.objects[name] = obj

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
            # Revert contnode to plain literal
            contnode["classes"] = []
            return None

        # Set classes for valid cross reference to object
        domain_name = type(self).name

        contnode["classes"] = ["xref", domain_name, f"{domain_name}-param"]

        return make_refnode(
            builder=builder,
            fromdocname=fromdocname,
            todocname=obj.docname,
            targetid=obj.node_id,
            child=contnode,
            title=target,
        )

    def get_objects(self) -> Iterator[tuple[str, str, str, str, str, int]]:
        for obj_key, obj in self.objects.items():
            # name: Fully qualified name.
            name: str = obj_key
            # dispname: Name to display when searching/linking.
            dispname: str = obj_key
            # type_: Object type, a key in ``self.object_types``.
            type_: str = "param"
            # docname: The document where it is to be found.
            docname: str = obj.docname
            # anchor: The anchor name for the object.
            anchor: str = obj.node_id
            # priority: How "important" the object is.
            #   Determines placement in search results. One of:
            #   1: Default priority (placed before full-text matches).
            #   0: Object is important (placed before default-priority objects).
            #   2: Object is unimportant (placed after full-text matches).
            #  -1: Object should not show up in search at all.
            priority: Literal[-1, 0, 1, 2] = 0
            yield (name, dispname, type_, docname, anchor, priority)


class WarpXDomain(ParmParseDomain):
    label = "WarpX"


def warpx_source_read(app: Sphinx, docname: str, source: list[str]):
    # Add default-role defintion at beginning of each document.
    # This makes `XYZ` equivalent to :pp:param:`XYZ`.
    for i in range(len(source)):
        new_doc_txt = ".. default-role:: pp:param\n"
        new_doc_txt += "\n"
        new_doc_txt += source[i]
        source[i] = new_doc_txt

    # Replace double backticks with single backticks.
    #
    # This makes ``...`` effectively the same as the default role `...`.
    # This, along with setting the default role to :pp:param:, avoids having to
    # find and edit every instance of a parameter currently enclosed in double
    # backticks in the documentation files. This is also makes things more
    # convenient and consistent for future contributors to the docs.
    #
    # This is implemented by replacing the pattern "``XYZ``" with "\ `XYZ`".
    # The backslash-escaped whitespace is a hack to preserve the same string
    # length as ``XYZ`` while rendering the same as `XYZ`. The purpose is to
    # avoid breaking the RST tables, which are sensitive to whitespace
    # alignment. For example:
    #
    # Before:
    # =============  ==================================
    # ``q_e``        Elementary charge (C)
    # ``m_e``        Electron mass (kg)
    # ``m_p``        Proton mass (kg)
    # ``m_u``        Unified atomic mass unit (kg)
    # ...
    # =============  ==================================
    #
    # After:
    # =============  ==================================
    # \ `q_e`        Elementary charge (C)
    # \ `m_e`        Electron mass (kg)
    # \ `m_p`        Proton mass (kg)
    # \ `m_u`        Unified atomic mass unit (kg)
    # ...
    # =============  ==================================
    #
    # Backslash-escaped whitespace characters are removed from the document:
    # https://docutils.sourceforge.io/0.4/docs/ref/rst/restructuredtext.html#escaping-mechanism

    # Pattern for ``XYZ``
    literal_re = re.compile(r"``([^`]+)``", re.DOTALL)

    def _repl(m: re.Match[str]):
        # Replace r"``XYZ``" with r"\ `XYZ`"
        return r"\ `" + m.group(1) + r"`"

    for i in range(len(source)):
        source[i] = literal_re.sub(_repl, source[i])


def setup(app: Sphinx) -> dict:
    app.add_domain(WarpXDomain)
    app.connect("source-read", warpx_source_read)
    return {
        "version": "0.1.0",
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }
