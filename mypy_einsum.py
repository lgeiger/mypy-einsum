from mypy.plugin import Plugin, FunctionSigContext
from mypy.nodes import StrExpr
from mypy.errorcodes import ErrorCode

einsum_symbols = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
einsum_symbols_set = set(einsum_symbols)
einsum_function_names = {
    "numpy.core.einsumfunc.einsum",
    "numpy.core.einsumfunc.einsum_path",
    "jax.numpy.einsum",
    "jax.numpy.einsum_path",
}


# Validation from https://github.com/numpy/numpy/blob/907ccc3467006df46a95ef63f08c7ca546ff2c49/numpy/_core/einsumfunc.py#L552-L726
def _parse_einsum_input(subscripts: str, operands: list):
    subscripts = subscripts.replace(" ", "")
    # Ensure all characters are valid
    for s in subscripts:
        if s in ".,->":
            continue
        if s not in einsum_symbols:
            raise ValueError("Character %s is not a valid symbol." % s)

    # Check for proper "->"
    if ("-" in subscripts) or (">" in subscripts):
        invalid = (subscripts.count("-") > 1) or (subscripts.count(">") > 1)
        if invalid or (subscripts.count("->") != 1):
            raise ValueError("Subscripts can only contain one '->'.")

    # Parse ellipses
    if "." in subscripts:
        used = subscripts.replace(".", "").replace(",", "").replace("->", "")
        unused = list(einsum_symbols_set - set(used))
        ellipse_inds = "".join(unused)
        longest = 0

        if "->" in subscripts:
            input_tmp, output_sub = subscripts.split("->")
            split_subscripts = input_tmp.split(",")
            out_sub = True
        else:
            split_subscripts = subscripts.split(",")
            out_sub = False

        for num, sub in enumerate(split_subscripts):
            if "." in sub:
                if (sub.count(".") != 3) or (sub.count("...") != 1):
                    raise ValueError("Invalid Ellipses.")

        subscripts = ",".join(split_subscripts)
        if longest == 0:
            out_ellipse = ""
        else:
            out_ellipse = ellipse_inds[-longest:]

        if out_sub:
            subscripts += "->" + output_sub.replace("...", out_ellipse)
        else:
            # Special care for outputless ellipses
            output_subscript = ""
            tmp_subscripts = subscripts.replace(",", "")
            for s in sorted(set(tmp_subscripts)):
                if s not in (einsum_symbols):
                    raise ValueError("Character %s is not a valid symbol." % s)
                if tmp_subscripts.count(s) == 1:
                    output_subscript += s
            normal_inds = "".join(sorted(set(output_subscript) - set(out_ellipse)))

            subscripts += "->" + out_ellipse + normal_inds

    # Build output string if does not exist
    if "->" in subscripts:
        input_subscripts, output_subscript = subscripts.split("->")
    else:
        input_subscripts = subscripts
        # Build output subscripts
        tmp_subscripts = subscripts.replace(",", "")
        output_subscript = ""
        for s in sorted(set(tmp_subscripts)):
            if s not in einsum_symbols:
                raise ValueError("Character %s is not a valid symbol." % s)
            if tmp_subscripts.count(s) == 1:
                output_subscript += s

    # Make sure output subscripts are in the input
    for char in output_subscript:
        if char not in input_subscripts:
            raise ValueError("Output character %s did not appear in the input" % char)

    # Make sure number operands is equivalent to the number of terms
    if len(input_subscripts.split(",")) != len(operands):
        raise ValueError(
            "Number of einsum subscripts must be equal to the " "number of operands."
        )


EINSUM = ErrorCode("einsum", "Check that Einsum notation is valid", "Einsum")


def einsum_callback(ctx: FunctionSigContext):
    subscripts_arg = ctx.args[0][0]
    if isinstance(subscripts_arg, StrExpr):
        try:
            _parse_einsum_input(subscripts_arg.value, ctx.args[1])
        except ValueError as e:
            ctx.context.set_line(subscripts_arg)
            ctx.api.fail(e.args[0], ctx.context, code=EINSUM)

    return ctx.default_signature


class EinsumPlugin(Plugin):
    def get_function_signature_hook(self, fullname: str):
        if fullname in einsum_function_names:
            return einsum_callback
        return None


def plugin(version: str):
    return EinsumPlugin
