import * as ts from "typescript";
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
    diagnostics: readonly ts.Diagnostic[],
    syntaxDiag: readonly ts.Diagnostic[]
): readonly ts.Diagnostic[] {

    const parseErrors = new Set();
    for (const e of syntaxDiag) {
        parseErrors.add(e);
    }
    const typecheckDiag = diagnostics.filter(e => !parseErrors.has(e));

    return typecheckDiag;
}

function main() {
    // Code adapted from:
    // https://github.com/GammaTauAI/opentau/blob/main/ts-compiler/main.ts
    // https://github.com/GammaTauAI/opentau/blob/main/evaluator/scripts/ts-does-parse/main.ts
    //
    // The ts-morph library is easier to work with than the typescript compiler
    // API, but is also much slower.

    const args = parseArgs();
    const compilerOptions = {
        target: ts.ScriptTarget.Latest,
        module: ts.ModuleKind.CommonJS,
        strict: false,
        noEmit: true,
    };

    const defaultCompilerHost = ts.createCompilerHost(compilerOptions);

    const makeCompilerHost = (
        filename: string,
        sourceFile: ts.SourceFile
    ): ts.CompilerHost => ({
        getSourceFile: (name, languageVersion) => {
            if (name === filename) {
                return sourceFile;
            } else {
                return defaultCompilerHost.getSourceFile(name, languageVersion);
            }
        },
        writeFile: (_filename, _data) => {},
        getDefaultLibFileName: () =>
            defaultCompilerHost.getDefaultLibFileName(compilerOptions),
        useCaseSensitiveFileNames: () => false,
        getCanonicalFileName: (filename) => filename,
        getCurrentDirectory: () => "",
        getNewLine: () => "\n",
        getDirectories: () => [],
        fileExists: () => true,
        readFile: () => "",
    });

    const createProgram = (code: string, setParentNodes = false): ts.Program => {
        const prog = ts.createProgram({
            rootNames: ["file.ts"],
            options: compilerOptions,
            host: makeCompilerHost(
                "file.ts",
                ts.createSourceFile(
                    "file.ts",
                    code,
                    ts.ScriptTarget.Latest,
                    setParentNodes,
                    ts.ScriptKind.TS
                )
            ),
        });
        return prog;
    };

    let buffer = "";
    process.stdin.on("data", (chunk) => {
        buffer = buffer.concat(chunk.toString());
    });

    process.stdin.on("close", () => {
        const program = createProgram(buffer);
        const file = program.getSourceFile("file.ts")!;
        const diagnostics = ts.getPreEmitDiagnostics(program, file);
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
