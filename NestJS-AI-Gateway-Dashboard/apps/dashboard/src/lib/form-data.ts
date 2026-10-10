/** Reads a text field from submitted form data; file inputs and missing fields yield ''. */
export function textField(formData: FormData, name: string): string {
  const value = formData.get(name);
  return typeof value === 'string' ? value.trim() : '';
}
