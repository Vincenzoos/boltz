# Boltz Binder Remodel Config UI

in Boltz_Remodel_UI.ipynb,
Instead of having user editing the raw json, abstract its into a more user-friendly interface (As a form with fields).

where user can define the target name and sequence,
set number of binders to remodel against the target,

then ui will generate binder sequence fields (name and sequence) for user to edit.

if number of binders too many, make sure the binder sequence fields are scrollable/paginated.

user can load existing binder remodel config json file to edit or create new one from it

binder/target name and sequence validation rules:
- name: must be a string, cannot be empty, cannot contain spaces, cannot contain special characters (except underscore and hyphen), cannot start with a number, max length 100 characters
- sequence: must be protein characters only (20 standard amino acids), cannot be empty, cannot contain spaces, in capital letters, max length 500. limit to sequence with length <= 500 residues. auto capitalize the sequence when user input.

add helpful text to UI for the target/binder name and sequence validation to help user avoid validation errors.