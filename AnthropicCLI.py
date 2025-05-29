import os
import re
import sys
from typing import Optional, List, Dict

import click
from anthropic import Anthropic, APIError


class ProjectContext:
    """Collect context information from a project directory."""

    DEFAULT_FILE_EXTENSIONS = [
        '.py', '.js', '.ts', '.html', '.css', '.md', '.txt', '.java',
        '.properties', '.gradle', '.xml', '.json', '.yaml', '.yml', '.toml' # Added common config/data files
    ]

    def __init__(self, project_dir: str, file_extensions: Optional[List[str]] = None):
        """Initialize with a project directory.

        Args:
            project_dir: Path to the project directory
            file_extensions: List of file extensions to include (e.g., ['.py', '.js'])
        """
        self.project_dir = os.path.abspath(project_dir)
        self.file_extensions = file_extensions if file_extensions is not None else self.DEFAULT_FILE_EXTENSIONS

        if not os.path.isdir(self.project_dir):
            raise ValueError(f"Project directory does not exist: {self.project_dir}")

    def get_project_structure(self) -> str:
        """Get a textual representation of the project structure.
        Optimized: Uses list comprehension and join for better string building.
        """
        structure_lines = [f"Project Structure ({self.project_dir}):"]

        for root, dirs, files in os.walk(self.project_dir):
            # Skip hidden directories like .git, .venv, node_modules etc.
            dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ['node_modules', '__pycache__', 'venv']]

            level = root.replace(self.project_dir, '').count(os.sep)
            indent = ' ' * 4 * level
            rel_path = os.path.relpath(root, self.project_dir)
            if rel_path != '.':
                structure_lines.append(f"{indent}{os.path.basename(root)}/")

            sub_indent = ' ' * 4 * (level + 1)
            for file in files:
                if any(file.endswith(ext) for ext in self.file_extensions):
                    structure_lines.append(f"{sub_indent}{file}")

        return "\n".join(structure_lines)

    def get_file_content(self, file_path: str) -> str:
        """Get the content of a specific file."""
        full_path = os.path.join(self.project_dir, file_path)
        try:
            with open(full_path, 'r', encoding='utf-8') as f:
                return f.read()
        except FileNotFoundError:
            return f"Error: File not found at {file_path}"
        except UnicodeDecodeError:
            return f"Error: Could not decode file content (not UTF-8) for {file_path}"
        except IOError as e:
            return f"Error reading file {file_path}: {str(e)}"

    def collect_project_files(self, max_files: int = 10, max_size_kb: int = 100) -> Dict[str, str]:
        """Collect content from project files, respecting size limits.
        Optimized: Uses os.walk for a single traversal instead of multiple glob calls.
                   More specific error handling for file reading.

        Args:
            max_files: Maximum number of files to include
            max_size_kb: Maximum size per file in KB

        Returns:
            Dictionary mapping relative file paths to their contents
        """
        files_dict = {}
        files_count = 0
        max_size_bytes = max_size_kb * 1024

        for root, _, files in os.walk(self.project_dir):
            # Skip hidden directories or common build/dependency folders
            if any(part.startswith('.') or part in ['node_modules', '__pycache__', 'venv', 'target', 'build', 'dist'] for part in os.path.relpath(root, self.project_dir).split(os.sep)):
                continue

            for file_name in files:
                if files_count >= max_files:
                    return files_dict # Early exit if max files reached

                if not any(file_name.endswith(ext) for ext in self.file_extensions):
                    continue # Skip if extension not matched

                file_path = os.path.join(root, file_name)

                try:
                    if os.path.getsize(file_path) > max_size_bytes:
                        click.echo(f"Skipping large file: {os.path.relpath(file_path, self.project_dir)} (>{max_size_kb}KB)", err=True)
                        continue
                except FileNotFoundError:
                    click.echo(f"Warning: File not found during size check: {os.path.relpath(file_path, self.project_dir)}", err=True)
                    continue
                except OSError as e: # Catch other OS-related errors like permission issues
                    click.echo(f"Warning: Could not get size for {os.path.relpath(file_path, self.project_dir)}: {e}", err=True)
                    continue

                rel_path = os.path.relpath(file_path, self.project_dir)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        files_dict[rel_path] = f.read()
                    files_count += 1
                except UnicodeDecodeError:
                    click.echo(f"Skipping non-text file: {rel_path} (not UTF-8)", err=True)
                except IOError as e:
                    click.echo(f"Skipping file due to read error: {rel_path} - {e}", err=True)
                except Exception as e: # Catch any other unexpected errors during read
                    click.echo(f"Skipping file due to unexpected error: {rel_path} - {e}", err=True)

        return files_dict

    def format_context(self, max_files: int = 10, max_size_kb: int = 100) -> str:
        """Format project context as a string for inclusion in prompts.

        Returns:
            A string containing project structure and key file contents
        """
        context_parts = []
        context_parts.append(self.get_project_structure())
        context_parts.append("\n\nKey files content:\n") # Changed to use same style for all parts

        files_dict = self.collect_project_files(max_files, max_size_kb)
        if not files_dict:
            context_parts.append("\nNo relevant files collected based on filters.")

        for file_path, content in files_dict.items():
            context_parts.append(f"\nFile: {file_path}\n")
            context_parts.append("```\n")
            context_parts.append(content)
            context_parts.append("\n```")

        return "".join(context_parts)


class PromptPlanExecutor:
    """Parses a plan file to extract individual prompts."""

    def __init__(self, plan_file: str):
        self.plan_file = plan_file
        self.prompts = self._parse_plan()

    def _parse_plan(self) -> list:
        """
        Parses the plan file to extract prompts enclosed in triple backticks.
        Optimized: More specific error handling for file operations.
        """
        try:
            with open(self.plan_file, 'r', encoding='utf-8') as f:
                content = f.read()

            # This handles optional language specifiers after the backticks
            prompts = re.findall(r'```(?:\w+)?\n(.*?)```', content, re.DOTALL)

            if not prompts:
                click.echo(f"Warning: No prompts found in {self.plan_file}. Ensure prompts are enclosed in ``` marks.", err=True)
                return []

            return [p.strip() for p in prompts]
        except FileNotFoundError:
            raise FileNotFoundError(f"Plan file not found: {self.plan_file}")
        except IOError as e:
            raise IOError(f"Error reading plan file {self.plan_file}: {str(e)}")
        except Exception as e: # Catch any other unexpected errors during parsing
            raise ValueError(f"Error parsing plan file {self.plan_file}: {str(e)}")


class AnthropicCLI:
    """CLI interface for interacting with the Anthropic Claude API."""

    DEFAULT_MODEL = "claude-sonnet-4-20250514" # Updated to a recent Sonnet model

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None, max_tokens: int = 1024):
        """Initialize the Anthropic client.
        Args:
            api_key: Anthropic API key (falls back to ANTHROPIC_API_KEY env var)
            model: Model to use for completions (defaults to claude-3-sonnet-20240229)
            max_tokens: Maximum tokens to generate in responses

        Raises:
            ValueError: If no API key is provided or found in environment
        """
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("Anthropic API key is required. Please provide it using the --api-key option or set the ANTHROPIC_API_KEY environment variable.")

        self.model = model or self.DEFAULT_MODEL
        self.max_tokens = max_tokens
        self.client = Anthropic(api_key=self.api_key)
        # New attributes to hold context and phase content for plan execution
        self.project_context_str_for_plan: str = ""
        self.phase_content_for_plan: str = ""

    def send_prompt(self, prompt: str) -> str:
        """Send a prompt to the Anthropic API and return the response.
        Optimized: More specific check for response content.

        Args:
            prompt: The text prompt to send

        Returns:
            The text response from Claude

        Raises:
            APIError: If the API request fails or response is malformed
        """
        try:
            message = self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                messages=[{"role": "user", "content": prompt}]
            )

            if not message or not hasattr(message, 'content') or not message.content:
                raise APIError("Received empty or malformed response from Anthropic API.")

            # Anthropic API content is a list of content blocks
            # We expect a single text block for basic text responses
            if isinstance(message.content, list) and message.content:
                text_content = ""
                for block in message.content:
                    if hasattr(block, 'text') and block.text is not None:
                        text_content += block.text
                if text_content:
                    return text_content
                else:
                    raise APIError("Received content without readable text from Anthropic API.")
            else:
                raise APIError("Unexpected content format from Anthropic API.")

        except APIError as e:
            # Re-raise Anthropic's own APIError
            raise e
        except Exception as e:
            # Catch any other unexpected network or client errors
            raise APIError(f"An unexpected error occurred during API request: {str(e)}")

    def execute_plan(self, plan_file: str) -> List[str]:
        """Execute a sequence of prompts from a plan file.
        Optimized: More specific error handling.
        Includes project context and phase content if set in the CLI instance.

        Args:
            plan_file: Path to the file containing the prompts

        Returns:
            List of responses from the API
        """
        try:
            executor = PromptPlanExecutor(plan_file)
        except (FileNotFoundError, IOError, ValueError) as e:
            click.echo(f"Error initializing plan executor: {e}", err=True)
            return []

        results: List[str] = []

        if not executor.prompts:
            click.echo("No prompts to execute in the plan file.", err=True)
            return []

        for i, prompt in enumerate(executor.prompts, 1):
            click.echo(f"\n{'='*10} Executing Step {i}/{len(executor.prompts)} {'='*10}")

            final_prompt_parts = []
            if self.project_context_str_for_plan:
                final_prompt_parts.append(f"PROJECT CONTEXT:\n{self.project_context_str_for_plan}\n")
            if self.phase_content_for_plan:
                final_prompt_parts.append(f"CURRENT PHASE GUIDANCE:\n{self.phase_content_for_plan}\n")

            final_prompt_parts.append(f"USER PROMPT FOR THIS STEP:\n{prompt}")

            full_prompt_to_send = "\n".join(final_prompt_parts)

            click.echo(f"Prompt (including context and phase guidance):\n{full_prompt_to_send}\n") # Show the full prompt being sent

            try:
                with click.progressbar(label="Waiting for API response", length=100) as bar: # Increased length for perceived progress
                    response = self.send_prompt(full_prompt_to_send)
                    bar.update(100) # Complete progress bar

                results.append(response)
                click.echo(f"\nResponse:\n{response}\n") # Added newline for better spacing

                if i < len(executor.prompts):
                    if not click.confirm("Continue to next step?"):
                        click.echo("Plan execution aborted by user.")
                        break
            except APIError as e:
                click.echo(f"API Error in step {i}: {str(e)}", err=True)
                if click.confirm("Continue to next step despite API error?"):
                    continue
                else:
                    click.echo("Plan execution aborted due to API error.")
                    break
            except Exception as e:
                click.echo(f"An unexpected error occurred in step {i}: {str(e)}", err=True)
                if click.confirm("Continue to next step despite unexpected error?"):
                    continue
                else:
                    click.echo("Plan execution aborted due to unexpected error.")
                    break

        return results


@click.command()
@click.option('--prompt', '-p', help='Single prompt to send to Anthropic API')
@click.option('--plan', '-f', type=click.Path(), default=None,
              help='Path to prompt plan file (defaults to prompt_plan.md in project-dir if not specified).')
@click.option('--api-key', '-k', help='Anthropic API Key (overrides environment variable)')
@click.option('--model', '-m', help=f'Model to use (default: {AnthropicCLI.DEFAULT_MODEL})')
@click.option('--max-tokens', type=int, default=1024, help='Maximum tokens in response (default: 1024)')
@click.option('--output', '-o', type=click.Path(), help='Save responses to this file')
@click.option('--project-dir', type=click.Path(exists=True, file_okay=False, dir_okay=True),
              help='Path to the project directory for context collection (optional)', default='.')
@click.option('--include-context', is_flag=True, help='Include project context (structure and key files) in the prompt.')
@click.option('--max-context-files', type=int, default=10, help='Max number of files to include in context.')
@click.option('--max-context-file-size-kb', type=int, default=100, help='Max size per file in KB for context.')
@click.option('--phase-file', type=click.Path(), default=None,
              help='Path to a phase guidance Markdown file (defaults to phase.md in project-dir if not specified).')
def main(prompt: Optional[str], plan: Optional[str], api_key: Optional[str],
         model: Optional[str], max_tokens: int, output: Optional[str],
         project_dir: str, include_context: bool, max_context_files: int, max_context_file_size_kb: int,
         phase_file: Optional[str]) -> None:
    """CLI tool for interacting with Anthropic's Claude API."""
    try:
        cli = AnthropicCLI(api_key=api_key, model=model, max_tokens=max_tokens)

        # --- Handle Project Context ---
        if include_context:
            try:
                project_context = ProjectContext(project_dir)
                click.echo(f"Collecting project context from: {project_dir}")
                context_str = project_context.format_context(max_files=max_context_files, max_size_kb=max_context_file_size_kb)
                cli.project_context_str_for_plan = context_str # Store for plan execution
                click.echo("Project context collected successfully.")
            except ValueError as e:
                click.echo(f"Error collecting project context: {e}", err=True)
                # If context is critical for the mode, exit; otherwise, proceed without context
                if not prompt and not plan: # If no explicit prompt or plan, assume context is main goal
                    sys.exit(1)
                click.echo("Proceeding without project context for now.")

        # --- Determine actual plan file path ---
        actual_plan_file: Optional[str] = None
        if plan: # User provided a specific path for --plan
            actual_plan_file = plan
        else: # Check for default prompt_plan.md in project root
            default_plan_path = os.path.join(project_dir, "prompt_plan.md")
            if os.path.exists(default_plan_path):
                actual_plan_file = default_plan_path
                click.echo(f"Using default prompt plan: {actual_plan_file}")
            # else: actual_plan_file remains None


        # --- Determine actual phase file path and load content ---
        actual_phase_file: Optional[str] = None
        if phase_file: # User provided a specific path for --phase-file
            actual_phase_file = phase_file
        else: # Check for default phase.md in project root
            default_phase_path = os.path.join(project_dir, "phase.md")
            if os.path.exists(default_phase_path):
                actual_phase_file = default_phase_path
                click.echo(f"Using default phase file: {actual_phase_file}")
            # else: actual_phase_file remains None

        if actual_phase_file:
            try:
                with open(actual_phase_file, 'r', encoding='utf-8') as f:
                    cli.phase_content_for_plan = f.read()
                click.echo(f"Phase guidance loaded from: {actual_phase_file}")
            except (FileNotFoundError, IOError) as e:
                click.echo(f"Error loading phase file {actual_phase_file}: {e}", err=True)
                # Do not exit here, just warn and proceed without phase guidance
                cli.phase_content_for_plan = "" # Ensure it's empty if error occurs

        # --- Execute based on prompt or plan ---
        if prompt:
            final_prompt_parts = []
            if cli.project_context_str_for_plan:
                final_prompt_parts.append(f"PROJECT CONTEXT:\n{cli.project_context_str_for_plan}\n")
            if cli.phase_content_for_plan:
                final_prompt_parts.append(f"CURRENT PHASE GUIDANCE:\n{cli.phase_content_for_plan}\n")
            final_prompt_parts.append(f"USER PROMPT:\n{prompt}")
            final_prompt = "\n".join(final_prompt_parts)

            click.echo("Sending prompt to Claude...")
            response = cli.send_prompt(final_prompt)
            click.echo(response)
            if output:
                try:
                    with open(output, 'w', encoding='utf-8') as f:
                        f.write(response)
                    click.echo(f"Response saved to {output}")
                except IOError as e:
                    click.echo(f"Error saving output to {output}: {e}", err=True)
        elif actual_plan_file: # Use actual_plan_file for plan execution
            responses = cli.execute_plan(actual_plan_file)
            if output:
                try:
                    with open(output, 'w', encoding='utf-8') as f:
                        for i, resp in enumerate(responses, 1):
                            f.write(f"--- Response {i} ---\n{resp}\n\n")
                    click.echo(f"All responses saved to {output}")
                except IOError as e:
                    click.echo(f"Error saving output to {output}: {e}", err=True)
        else:
            click.echo("Please provide either a prompt (-p), a plan file (-f), or ensure 'prompt_plan.md' exists in the project directory.", err=True)
            sys.exit(1)
    except ValueError as e:
        click.echo(f"Configuration Error: {str(e)}", err=True)
        sys.exit(1)
    except APIError as e:
        click.echo(f"API Error: {str(e)}", err=True)
        sys.exit(1)
    except Exception as e:
        click.echo(f"An unexpected error occurred: {str(e)}", err=True)
        sys.exit(1)


if __name__ == '__main__':
    main()