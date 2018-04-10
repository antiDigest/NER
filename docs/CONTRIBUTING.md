# How to contribute

Third-party patches are essential for keeping Puppet great. We simply can't
access the huge number of platforms and myriad configurations for running
Puppet. We want to keep it as easy as possible to contribute changes that
get things working in your environment. There are a few guidelines that we
need contributors to follow so that we can have a chance of keeping on
top of things.

## Getting Started

* Make sure you have a [GitHub account](https://github.com/signup/free).
* Submit an Issue for your issue if one does not already exist.
  * Clearly describe the issue including steps to reproduce when it is a bug.
  * Make sure you fill in the earliest version that you know has the issue.
  * An Issue is not necessary for trivial changes.
* Fork the repository on GitHub.

## Making Changes

* Create a topic branch from where you want to base your work.
  * This is usually the master branch.
  * Only target release branches if you are certain your fix must be on that
    branch.
  * To quickly create a topic branch based on master, run `git checkout -b
    fix/master/my_contribution master`. Please avoid working directly on the
    `master` branch.
* Make commits of logical and atomic units.
* Check for unnecessary whitespace with `git diff --check` before committing.
* Make sure your commit messages are in the proper format. If the commit
  addresses an issue, start the first line of the commit with the issue number
  in parentheses.
* Make sure you have added the necessary tests for your changes.
* Run _all_ the tests to assure nothing else was accidentally broken. First
  install all the test dependencies with `bundle install --path .bundle`. Then
  either run all the tests serially with `bundle exec rspec spec` or in parallel
  with `bundle exec rake parallel:spec[process_count]`

## Writing Translatable Code

When adding user-facing strings to your work, follow these guidelines:

* Use full sentences. Strings built up out of concatenated bits are hard to translate.
* Use string formatting instead of interpolation. Use the hash format and give good names to the placeholder values that can be used by translators to understand the meaning of the formatted values.
  For example: `_('Creating new user %{name}.') % { name: user.name }`
* Use `n_()` for pluralization. (see gettext gem docs linked above for details)

It is the responsibility of contributors and code reviewers to ensure that all
user-facing strings are marked in new PRs before merging.

## Making Trivial Changes

For [changes of a trivial nature](https://docs.puppet.com/community/trivial_patch_exemption.html), it is not always necessary to create a new
ticket in Jira. In this case, it is appropriate to start the first line of a
commit with one of  `(docs)`, `(maint)`, or `(packaging)` instead of a ticket
number.

For commits that address trivial repository maintenance tasks or packaging
issues, start the first line of the commit with `(maint)` or `(packaging)`,
respectively.

## Submitting Changes

* Push your changes to a topic branch in your fork of the repository.
* Submit a pull request to the main repository.
* Mark that you have submitted code and are ready for it to be reviewed
  (Status: Ready for Merge).
  * Include a link to the pull request in the ticket.
* The core team looks at Pull Requests on a regular basis.
* After feedback has been given we expect responses within two weeks. After two
  weeks we may close the pull request if it isn't showing any activity.

## Revert Policy

By running tests in advance and by engaging with peer review for prospective
changes, your contributions have a high probability of becoming long lived
parts of the the project. After being merged, the code will run through a
series of testing pipelines on a large number of operating system
environments. These pipelines can reveal incompatibilities that are difficult
to detect in advance.

If the code change results in a test failure, we will make our best effort to
correct the error. If a fix cannot be determined and committed within 24 hours
of its discovery, the commit(s) responsible _may_ be reverted, at the
discretion of the committer and Puppet maintainers. This action would be taken
to help maintain passing states in our testing pipelines.

The original contributor will be notified of the revert in the Pull Request
associated with the change. A reference to the test(s) and operating system(s)
that failed as a result of the code change will also be added to the Pull Request.
This test(s) should be used to check future submissions of the code to
ensure the issue has been resolved.

### Summary

* Changes resulting in test pipeline failures will be reverted if they cannot
  be resolved within one business day.
