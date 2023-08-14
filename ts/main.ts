import { Diagnostic, DiagnosticWithLocation, Project, ts } from "ts-morph";
import yargs from "yargs";
import { hideBin } from "yargs/helpers";

interface Arguments {
    debug: boolean;
    $0: string,
    _: (string | number)[]
    [x: string]: unknown
}

function parseArgs(): Arguments {
    const yargsBuilder = yargs(hideBin(process.argv))
        .usage("Usage: $0")
        .option("debug", {
            alias: "d",
            describe: "enable debug output",
            type: "boolean",
            default: false
        })
        .help("help");

    const a = yargsBuilder.parseSync();

    return yargsBuilder.parseSync();
}

function filter_diagnostics(
    diagnostics: Diagnostic<ts.Diagnostic>[],
    syntaxDiag: DiagnosticWithLocation[],
): Diagnostic<ts.Diagnostic>[] {

    const parseErrors = new Set();
    for (const e of syntaxDiag) {
        parseErrors.add(e);
    }
    const typecheckDiag = diagnostics.filter(e => !parseErrors.has(e));

    return typecheckDiag;
}

function main() {
    const args = parseArgs();
    const project = new Project({
        useInMemoryFileSystem: true,
        compilerOptions: {
            target: ts.ScriptTarget.Latest,
            module: ts.ModuleKind.CommonJS,
            strict: false,
            noEmit: true,
        },
    });

    let buffer = "";
    process.stdin.on("data", (chunk) => {
        buffer = buffer.concat(chunk.toString());
    });

    process.stdin.on("close", () => {
        const file = project.createSourceFile("file.ts", buffer);
        const program = project.getProgram();
        const diagnostics = file.getPreEmitDiagnostics();
        const syntaxDiag = program.getSyntacticDiagnostics(file);

        // PreEmitDiagnostics includes all errors, so remove parse errors
        const typecheckDiag = filter_diagnostics(diagnostics, syntaxDiag);

        if (args.debug) {
            console.log("===Type checking errors===");
            console.log(typecheckDiag);
            console.log("===Parse errors===");
            console.log(syntaxDiag);
        }

        const result = JSON.stringify({
            type_errors: typecheckDiag.length,
            parse_errors: syntaxDiag.length,
        });
        console.log(result);
    });
}

main();
