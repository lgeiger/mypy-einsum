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
    "torch.functional.einsum",
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

                split_subscripts[num] = sub.replace("...", "")

        subscripts = ",".join(split_subscripts)

        if out_sub:
            subscripts += "->" + output_sub.replace("...", "")
        else:
            # Special care for outputless ellipses
            output_subscript = ""
            tmp_subscripts = subscripts.replace(",", "")
            for s in sorted(set(tmp_subscripts)):
                if s not in einsum_symbols:
                    raise ValueError("Character %s is not a valid symbol." % s)
                if tmp_subscripts.count(s) == 1:
                    output_subscript += s
            normal_inds = "".join(sorted(set(output_subscript)))

            subscripts += "->" + normal_inds

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

    # Make sure output subscripts are unique and in the input
    for char in output_subscript:
        if output_subscript.count(char) != 1:
            raise ValueError("Output character %s appeared multiple times." % char)
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
    if len(ctx.args) == 1:
        operand_args = ctx.args[0][1:]
    elif (
        ctx.default_signature.definition is not None
        and ctx.default_signature.definition.fullname == "jax.numpy.einsum"
        and len(ctx.args[1]) == 1
        and ctx.default_signature.arg_names[:2] == [None, None]
    ):
        # Handle mismatched overload: https://github.com/google/jax/blob/49c80e68d105dc93e5f26ef15b434b279bf00a03/jax/_src/numpy/lax_numpy.py#L3376-L3386
        operand_args = ctx.args[1] + ctx.args[2]
    else:
        operand_args = ctx.args[1]
    if isinstance(subscripts_arg, StrExpr):
        try:
            _parse_einsum_input(subscripts_arg.value, operand_args)
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
